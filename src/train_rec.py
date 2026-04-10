import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import transformers
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from loguru import logger
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel

from config import gpt2_special_tokens_dict, prompt_special_tokens_dict
from dataset_dbpedia import DBpedia
from dataset_rec import CRSRecDataset, CRSRecDataCollator, build_prompt_with_fiup  # [FIUP-NEW] build_prompt_with_fiup
from evaluate_rec import RecEvaluator
from model_gpt2 import PromptGPT2forCRS
from model_prompt import KGPrompt
from modules.sentiment import SentimentAnalyzer    # [FIUP-NEW]
from modules.fiup_manager import FIUPManager        # [FIUP-NEW]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--output_dir", type=str, default='save', help="Where to store the final model.")
    parser.add_argument("--debug", action='store_true', help="Debug mode.")
    # data
    parser.add_argument("--dataset", type=str, required=True, help="A file containing all data.")
    parser.add_argument("--shot", type=float, default=1)
    parser.add_argument("--use_resp", action="store_true")
    parser.add_argument("--context_max_length", type=int, help="max input length in dataset.")
    parser.add_argument("--prompt_max_length", type=int)
    parser.add_argument("--entity_max_length", type=int, help="max entity length in dataset.")
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--text_tokenizer", type=str)
    # model
    parser.add_argument("--model", type=str, required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--text_encoder", type=str)
    parser.add_argument("--num_bases", type=int, default=8, help="num_bases in RGCN.")
    parser.add_argument("--n_prefix_rec", type=int)
    parser.add_argument("--prompt_encoder", type=str)
#画像
    parser.add_argument("--use_fiup", action="store_true", default=False,
                        help="启用 FIUP 双库用户画像模块")
    parser.add_argument("--fiup_alpha", type=float, default=0.8,
                        help="FIUP 遗忘衰减系数")
    parser.add_argument("--fiup_lambda", type=float, default=0.1,
                        help="FIUP 重排序融合权重")
    parser.add_argument("--sentiment_backend", type=str, default="textblob",
                        choices=["textblob", "transformers"],
                        help="情感分析后端")

    # optim
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                        help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument('--max_grad_norm', type=float)
    parser.add_argument('--num_warmup_steps', type=int)
    parser.add_argument('--fp16', action='store_true')
    # wandb
    parser.add_argument("--use_wandb", action="store_true", help="whether to use wandb")
    parser.add_argument("--entity", type=str, help="wandb username")
    parser.add_argument("--project", type=str, help="wandb exp project")
    parser.add_argument("--name", type=str, help="wandb exp name")
    parser.add_argument("--log_all", action="store_true", help="log in all processes, otherwise only in rank0")

    args = parser.parse_args()
    return args


# ── [FIUP-NEW] 辅助函数：用 text_encoder 为单条文本提取 [CLS] 语义向量 ────────
def encode_context_emb(
    ctx_str: str,
    text_tokenizer,
    text_encoder,
    device,
    max_length: int = 128,
) -> torch.Tensor:
    """
    利用已有的 text_encoder（RoBERTa，frozen）为对话文本生成 [CLS] 向量。

    返回形状 [hidden_size] 的 CPU 张量，方便存入 FIUPManager。
    text_encoder 的 requires_grad 已被设为 False，此处无梯度开销。
    """
    if not ctx_str:
        return torch.zeros(text_encoder.config.hidden_size)

    tokenized = text_tokenizer(
        ctx_str,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = text_encoder(**tokenized)
        # last_hidden_state[:, 0, :] 即 [CLS] token 表示
        context_emb = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()  # [hidden_size]

    return context_emb


if __name__ == '__main__':
    args = parse_args()
    config = vars(args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us.
    mixed_precision = "fp16" if args.fp16 else "no"
    accelerator = Accelerator(device_placement=False, mixed_precision=mixed_precision)
    device = accelerator.device

    # Make one log on every process with the configuration for debugging.
    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    logger.remove()
    logger.add(sys.stderr, level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.add(f'log/{local_time}.log', level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.info(accelerator.state)
    logger.info(config)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    # wandb
    if args.use_wandb:
        name = args.name if args.name else local_time
        name += '_' + str(accelerator.process_index)

        if args.log_all:
            group = args.name if args.name else 'DDP_' + local_time
            run = wandb.init(entity=args.entity, project=args.project, group=group, config=config, name=name)
        else:
            if accelerator.is_local_main_process:
                run = wandb.init(entity=args.entity, project=args.project, config=config, name=name)
            else:
                run = None
    else:
        run = None

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    kg = DBpedia(dataset=args.dataset, debug=args.debug).get_entity_kg_info()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.add_special_tokens(gpt2_special_tokens_dict)
    model = PromptGPT2forCRS.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)

    text_tokenizer = AutoTokenizer.from_pretrained(args.text_tokenizer)
    text_tokenizer.add_special_tokens(prompt_special_tokens_dict)
    text_encoder = AutoModel.from_pretrained(args.text_encoder)
    text_encoder.resize_token_embeddings(len(text_tokenizer))
    text_encoder = text_encoder.to(device)

    prompt_encoder = KGPrompt(
        model.config.n_embd, text_encoder.config.hidden_size, model.config.n_head, model.config.n_layer, 2,
        n_entity=kg['num_entities'], num_relations=kg['num_relations'], num_bases=args.num_bases,
        edge_index=kg['edge_index'], edge_type=kg['edge_type'],
        n_prefix_rec=args.n_prefix_rec
    )
    if args.prompt_encoder is not None:
        prompt_encoder.load(args.prompt_encoder)
    prompt_encoder = prompt_encoder.to(device)

    fix_modules = [model, text_encoder]
    for module in fix_modules:
        module.requires_grad_(False)

    # optim & amp
    modules = [prompt_encoder]
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for model in modules for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for model in modules for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # data
    train_dataset = CRSRecDataset(
        dataset=args.dataset, split='train', debug=args.debug,
        tokenizer=tokenizer, context_max_length=args.context_max_length, use_resp=args.use_resp,
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
        entity_max_length=args.entity_max_length,
    )
    shot_len = int(len(train_dataset) * args.shot)
    train_dataset = random_split(train_dataset, [shot_len, len(train_dataset) - shot_len])[0]
    assert len(train_dataset) == shot_len
    valid_dataset = CRSRecDataset(
        dataset=args.dataset, split='valid', debug=args.debug,
        tokenizer=tokenizer, context_max_length=args.context_max_length, use_resp=args.use_resp,
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
        entity_max_length=args.entity_max_length,
    )
    test_dataset = CRSRecDataset(
        dataset=args.dataset, split='test', debug=args.debug,
        tokenizer=tokenizer, context_max_length=args.context_max_length, use_resp=args.use_resp,
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
        entity_max_length=args.entity_max_length,
    )
    data_collator = CRSRecDataCollator(
        tokenizer=tokenizer, device=device, debug=args.debug,
        context_max_length=args.context_max_length, entity_max_length=args.entity_max_length,
        pad_entity_id=kg['pad_entity_id'],
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        collate_fn=data_collator,
        shuffle=True
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=data_collator,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=data_collator,
    )
    evaluator = RecEvaluator()
    prompt_encoder, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
        prompt_encoder, optimizer, train_dataloader, valid_dataloader, test_dataloader
    )

    # step, epoch, batch size
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    completed_steps = 0

    # lr_scheduler
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, args.num_warmup_steps, args.max_train_steps)
    lr_scheduler = accelerator.prepare(lr_scheduler)

    # training info
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

    # save model with best metric
    metric, mode = 'loss', -1
    assert mode in (-1, 1)
    best_metric = 0 if mode == 1 else float('inf')
    best_metric_dir = os.path.join(args.output_dir, 'best')
    os.makedirs(best_metric_dir, exist_ok=True)

    # ── [FIUP-NEW] 初始化情感分析器和用户画像管理器 ──────────────────────────
    if args.use_fiup:
        sentiment_analyzer = SentimentAnalyzer(backend=args.sentiment_backend)
        fiup_managers: dict = {}
        EMB_DIM = text_encoder.config.hidden_size
        logger.info(f"[FIUP] Initialized. EMB_DIM={EMB_DIM}, alpha={args.fiup_alpha}, backend={args.sentiment_backend}")
    # ────────────────────────────────────────────────────────────────────────

    # train loop
    for epoch in range(args.num_train_epochs):
        train_loss = []
        prompt_encoder.train()
        for step, batch in enumerate(train_dataloader):

            # ── [FIUP] 开始 FIUP 注入（仅在 --use_fiup 时启用）─────────────────
            if args.use_fiup:
                batch_size         = len(batch["context"]["input_ids"])
                user_ids           = batch.get("user_id",       [f"u{step}_{i}" for i in range(batch_size)])
                context_strs       = batch.get("context_str",   [""] * batch_size)
                entity_names_batch = batch.get("entity_names",  [[] for _ in range(batch_size)])

                prompt_context_list = batch.get("_prompt_context_list", context_strs)
                prompt_kg_list      = batch.get("_prompt_kg_list",      [""] * batch_size)

                fiup_prompts = []
                for uid, ctx_str, attrs in zip(user_ids, context_strs, entity_names_batch):
                    if uid not in fiup_managers:
                        fiup_managers[uid] = FIUPManager(emb_dim=EMB_DIM, alpha=args.fiup_alpha)
                    mgr = fiup_managers[uid]

                    e_tau = sentiment_analyzer.score(ctx_str)

                    context_emb = encode_context_emb(
                        ctx_str, text_tokenizer, text_encoder, device, max_length=128
                    )

                    mgr.update_profile(attrs, e_tau, context_emb)
                    fiup_prompts.append(mgr.build_profile_prompt())

                new_prompt_encodings = [
                    build_prompt_with_fiup(
                        context_text=ctx,
                        kg_text=kg_text,
                        fiup_prompt=fp,
                        tokenizer=text_tokenizer,
                        max_length=args.prompt_max_length,
                    )
                    for ctx, kg_text, fp in zip(prompt_context_list, prompt_kg_list, fiup_prompts)
                ]
                batch['prompt'] = {
                    k: torch.cat([enc[k] for enc in new_prompt_encodings], dim=0).to(device)
                    for k in new_prompt_encodings[0].keys()
                }
                batch["fiup_prompts"] = fiup_prompts
            # ── [FIUP] 结束 FIUP 注入 ────────────────────────────────────────

            with torch.no_grad():
                token_embeds = text_encoder(**batch['prompt']).last_hidden_state
            # ── [FIUP-NEW] 结束 FIUP 注入 ────────────────────────────────────

            with torch.no_grad():
                token_embeds = text_encoder(**batch['prompt']).last_hidden_state
            prompt_embeds = prompt_encoder(
                entity_ids=batch['entity'],
                token_embeds=token_embeds,
                output_entity=True,
                use_rec_prefix=True
            )
            batch['context']['prompt_embeds'] = prompt_embeds
            batch['context']['entity_embeds'] = prompt_encoder.get_entity_embeds()

            loss = model(**batch['context'], rec=True).rec_loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            train_loss.append(float(loss))

            # optim step
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                if args.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(prompt_encoder.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                completed_steps += 1
                if run:
                    run.log({'loss': np.mean(train_loss) * args.gradient_accumulation_steps})

            if completed_steps >= args.max_train_steps:
                break

        # metric
        train_loss = np.mean(train_loss) * args.gradient_accumulation_steps
        logger.info(f'epoch {epoch} train loss {train_loss}')

        del train_loss, batch

        # ── valid ────────────────────────────────────────────────────────────
        valid_loss = []
        prompt_encoder.eval()
        for batch in tqdm(valid_dataloader):
            with torch.no_grad():
                token_embeds = text_encoder(**batch['prompt']).last_hidden_state
                prompt_embeds = prompt_encoder(
                    entity_ids=batch['entity'],
                    token_embeds=token_embeds,
                    output_entity=True,
                    use_rec_prefix=True
                )
                batch['context']['prompt_embeds'] = prompt_embeds
                batch['context']['entity_embeds'] = prompt_encoder.get_entity_embeds()

                outputs = model(**batch['context'], rec=True)
                valid_loss.append(float(outputs.rec_loss))
                logits = outputs.rec_logits[:, kg['item_ids']]
                ranks = torch.topk(logits, k=50, dim=-1).indices.tolist()
                ranks = [[kg['item_ids'][rank] for rank in batch_rank] for batch_rank in ranks]
                labels = batch['context']['rec_labels']
                evaluator.evaluate(ranks, labels)

        # metric
        report = accelerator.gather(evaluator.report())
        for k, v in report.items():
            report[k] = v.sum().item()

        valid_report = {}
        for k, v in report.items():
            if k != 'count':
                valid_report[f'valid/{k}'] = v / report['count']
        valid_report['valid/loss'] = np.mean(valid_loss)
        valid_report['epoch'] = epoch
        logger.info(f'{valid_report}')
        if run:
            run.log(valid_report)
        evaluator.reset_metric()

        if valid_report[f'valid/{metric}'] * mode > best_metric * mode:
            prompt_encoder.save(best_metric_dir)
            best_metric = valid_report[f'valid/{metric}']
            logger.info(f'new best model with {metric}')

        # ── test ─────────────────────────────────────────────────────────────
        test_loss = []
        prompt_encoder.eval()
        for batch in tqdm(test_dataloader):
            with torch.no_grad():
                token_embeds = text_encoder(**batch['prompt']).last_hidden_state
                prompt_embeds = prompt_encoder(
                    entity_ids=batch['entity'],
                    token_embeds=token_embeds,
                    output_entity=True,
                    use_rec_prefix=True
                )
                batch['context']['prompt_embeds'] = prompt_embeds
                batch['context']['entity_embeds'] = prompt_encoder.get_entity_embeds()

                outputs = model(**batch['context'], rec=True)
                test_loss.append(float(outputs.rec_loss))
                logits = outputs.rec_logits[:, kg['item_ids']]

                # ── [FIUP-NEW] 可选：显性库重排序 hook ───────────────────────
                # 启用方式：实现 get_batch_fiup_rerank() 并取消注释：
                #
                # logits = get_batch_fiup_rerank(
                #     logits=logits,
                #     item_ids=kg['item_ids'],
                #     batch=batch,
                #     fiup_managers=fiup_managers,
                #     fiup_lambda=0.1,
                # )
                #
                # get_batch_fiup_rerank 参考实现见文档"可选修改：推荐重排序"章节
                # ─────────────────────────────────────────────────────────────

                ranks = torch.topk(logits, k=50, dim=-1).indices.tolist()
                ranks = [[kg['item_ids'][rank] for rank in batch_rank] for batch_rank in ranks]
                labels = batch['context']['rec_labels']
                evaluator.evaluate(ranks, labels)

        # metric
        report = accelerator.gather(evaluator.report())
        for k, v in report.items():
            report[k] = v.sum().item()

        test_report = {}
        for k, v in report.items():
            if k != 'count':
                test_report[f'test/{k}'] = v / report['count']
        test_report['test/loss'] = np.mean(test_loss)
        test_report['epoch'] = epoch
        logger.info(f'{test_report}')
        if run:
            run.log(test_report)
        evaluator.reset_metric()

    final_dir = os.path.join(args.output_dir, 'final')
    prompt_encoder.save(final_dir)
    logger.info(f'save final model')



# import argparse
# import math
# import os
# import sys
# import time

# import numpy as np
# import torch
# import transformers
# import wandb
# from accelerate import Accelerator
# from accelerate.utils import set_seed
# from loguru import logger
# from torch.utils.data import DataLoader, random_split
# from tqdm.auto import tqdm
# from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel

# from config import gpt2_special_tokens_dict, prompt_special_tokens_dict
# from dataset_dbpedia import DBpedia
# from dataset_rec import CRSRecDataset, CRSRecDataCollator
# from evaluate_rec import RecEvaluator
# from model_gpt2 import PromptGPT2forCRS
# from model_prompt import KGPrompt


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
#     parser.add_argument("--output_dir", type=str, default='save', help="Where to store the final model.")
#     parser.add_argument("--debug", action='store_true', help="Debug mode.")
#     # data
#     parser.add_argument("--dataset", type=str, required=True, help="A file containing all data.")
#     parser.add_argument("--shot", type=float, default=1)
#     parser.add_argument("--use_resp", action="store_true")
#     parser.add_argument("--context_max_length", type=int, help="max input length in dataset.")
#     parser.add_argument("--prompt_max_length", type=int)
#     parser.add_argument("--entity_max_length", type=int, help="max entity length in dataset.")
#     parser.add_argument('--num_workers', type=int, default=0)
#     parser.add_argument("--tokenizer", type=str)
#     parser.add_argument("--text_tokenizer", type=str)
#     # model
#     parser.add_argument("--model", type=str, required=True,
#                         help="Path to pretrained model or model identifier from huggingface.co/models.")
#     parser.add_argument("--text_encoder", type=str)
#     parser.add_argument("--num_bases", type=int, default=8, help="num_bases in RGCN.")
#     parser.add_argument("--n_prefix_rec", type=int)
#     parser.add_argument("--prompt_encoder", type=str)
#     # optim
#     parser.add_argument("--num_train_epochs", type=int, default=10, help="Total number of training epochs to perform.")
#     parser.add_argument("--max_train_steps", type=int, default=None,
#                         help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
#     parser.add_argument("--per_device_train_batch_size", type=int, default=4,
#                         help="Batch size (per device) for the training dataloader.")
#     parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
#                         help="Batch size (per device) for the evaluation dataloader.")
#     parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
#                         help="Number of updates steps to accumulate before performing a backward/update pass.")
#     parser.add_argument("--learning_rate", type=float, default=1e-5,
#                         help="Initial learning rate (after the potential warmup period) to use.")
#     parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
#     parser.add_argument('--max_grad_norm', type=float)
#     parser.add_argument('--num_warmup_steps', type=int)
#     parser.add_argument('--fp16', action='store_true')
#     # wandb
#     parser.add_argument("--use_wandb", action="store_true", help="whether to use wandb")
#     parser.add_argument("--entity", type=str, help="wandb username")
#     parser.add_argument("--project", type=str, help="wandb exp project")
#     parser.add_argument("--name", type=str, help="wandb exp name")
#     parser.add_argument("--log_all", action="store_true", help="log in all processes, otherwise only in rank0")

#     args = parser.parse_args()
#     return args


# if __name__ == '__main__':
#     args = parse_args()
#     config = vars(args)

#     # Initialize the accelerator. We will let the accelerator handle device placement for us.
#     # accelerator = Accelerator(device_placement=False, fp16=args.fp16)
#     mixed_precision = "fp16" if args.fp16 else "no"
#     accelerator = Accelerator(device_placement=False, mixed_precision=mixed_precision)
#     device = accelerator.device

#     # Make one log on every process with the configuration for debugging.
#     local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
#     logger.remove()
#     logger.add(sys.stderr, level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
#     logger.add(f'log/{local_time}.log', level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
#     logger.info(accelerator.state)
#     logger.info(config)

#     if accelerator.is_local_main_process:
#         transformers.utils.logging.set_verbosity_info()
#     else:
#         transformers.utils.logging.set_verbosity_error()
#     # wandb
#     if args.use_wandb:
#         name = args.name if args.name else local_time
#         name += '_' + str(accelerator.process_index)

#         if args.log_all:
#             group = args.name if args.name else 'DDP_' + local_time
#             run = wandb.init(entity=args.entity, project=args.project, group=group, config=config, name=name)
#         else:
#             if accelerator.is_local_main_process:
#                 run = wandb.init(entity=args.entity, project=args.project, config=config, name=name)
#             else:
#                 run = None
#     else:
#         run = None

#     # If passed along, set the training seed now.
#     if args.seed is not None:
#         set_seed(args.seed)

#     if args.output_dir is not None:
#         os.makedirs(args.output_dir, exist_ok=True)

#     kg = DBpedia(dataset=args.dataset, debug=args.debug).get_entity_kg_info()

#     tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
#     tokenizer.add_special_tokens(gpt2_special_tokens_dict)
#     model = PromptGPT2forCRS.from_pretrained(args.model)
#     model.resize_token_embeddings(len(tokenizer))
#     model.config.pad_token_id = tokenizer.pad_token_id
#     model = model.to(device)

#     text_tokenizer = AutoTokenizer.from_pretrained(args.text_tokenizer)
#     text_tokenizer.add_special_tokens(prompt_special_tokens_dict)
#     text_encoder = AutoModel.from_pretrained(args.text_encoder)
#     text_encoder.resize_token_embeddings(len(text_tokenizer))
#     text_encoder = text_encoder.to(device)

#     prompt_encoder = KGPrompt(
#         model.config.n_embd, text_encoder.config.hidden_size, model.config.n_head, model.config.n_layer, 2,
#         n_entity=kg['num_entities'], num_relations=kg['num_relations'], num_bases=args.num_bases,
#         edge_index=kg['edge_index'], edge_type=kg['edge_type'],
#         n_prefix_rec=args.n_prefix_rec
#     )
#     if args.prompt_encoder is not None:
#         prompt_encoder.load(args.prompt_encoder)
#     prompt_encoder = prompt_encoder.to(device)

#     fix_modules = [model, text_encoder]
#     for module in fix_modules:
#         module.requires_grad_(False)

#     # optim & amp
#     modules = [prompt_encoder]
#     no_decay = ["bias", "LayerNorm.weight"]
#     optimizer_grouped_parameters = [
#         {
#             "params": [p for model in modules for n, p in model.named_parameters()
#                        if not any(nd in n for nd in no_decay) and p.requires_grad],
#             "weight_decay": args.weight_decay,
#         },
#         {
#             "params": [p for model in modules for n, p in model.named_parameters()
#                        if any(nd in n for nd in no_decay) and p.requires_grad],
#             "weight_decay": 0.0,
#         },
#     ]
#     optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
#     # data
#     train_dataset = CRSRecDataset(
#         dataset=args.dataset, split='train', debug=args.debug,
#         tokenizer=tokenizer, context_max_length=args.context_max_length, use_resp=args.use_resp,
#         prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
#         entity_max_length=args.entity_max_length,
#     )
#     shot_len = int(len(train_dataset) * args.shot)
#     train_dataset = random_split(train_dataset, [shot_len, len(train_dataset) - shot_len])[0]
#     assert len(train_dataset) == shot_len
#     valid_dataset = CRSRecDataset(
#         dataset=args.dataset, split='valid', debug=args.debug,
#         tokenizer=tokenizer, context_max_length=args.context_max_length, use_resp=args.use_resp,
#         prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
#         entity_max_length=args.entity_max_length,
#     )
#     test_dataset = CRSRecDataset(
#         dataset=args.dataset, split='test', debug=args.debug,
#         tokenizer=tokenizer, context_max_length=args.context_max_length, use_resp=args.use_resp,
#         prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
#         entity_max_length=args.entity_max_length,
#     )
#     data_collator = CRSRecDataCollator(
#         tokenizer=tokenizer, device=device, debug=args.debug,
#         context_max_length=args.context_max_length, entity_max_length=args.entity_max_length,
#         pad_entity_id=kg['pad_entity_id'],
#         prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
#     )
#     train_dataloader = DataLoader(
#         train_dataset,
#         batch_size=args.per_device_train_batch_size,
#         collate_fn=data_collator,
#         shuffle=True
#     )
#     valid_dataloader = DataLoader(
#         valid_dataset,
#         batch_size=args.per_device_eval_batch_size,
#         collate_fn=data_collator,
#     )
#     test_dataloader = DataLoader(
#         test_dataset,
#         batch_size=args.per_device_eval_batch_size,
#         collate_fn=data_collator,
#     )
#     evaluator = RecEvaluator()
#     prompt_encoder, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
#         prompt_encoder, optimizer, train_dataloader, valid_dataloader, test_dataloader
#     )
#     # step, epoch, batch size
#     num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
#     if args.max_train_steps is None:
#         args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
#     else:
#         args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
#     total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
#     completed_steps = 0
#     # lr_scheduler
#     lr_scheduler = get_linear_schedule_with_warmup(optimizer, args.num_warmup_steps, args.max_train_steps)
#     lr_scheduler = accelerator.prepare(lr_scheduler)
#     # training info
#     logger.info("***** Running training *****")
#     logger.info(f"  Num examples = {len(train_dataset)}")
#     logger.info(f"  Num Epochs = {args.num_train_epochs}")
#     logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
#     logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
#     logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
#     logger.info(f"  Total optimization steps = {args.max_train_steps}")
#     # Only show the progress bar once on each machine.
#     progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

#     # save model with best metric
#     metric, mode = 'loss', -1
#     assert mode in (-1, 1)
#     if mode == 1:
#         best_metric = 0
#     else:
#         best_metric = float('inf')
#     best_metric_dir = os.path.join(args.output_dir, 'best')
#     os.makedirs(best_metric_dir, exist_ok=True)

#     # train loop
#     for epoch in range(args.num_train_epochs):
#         train_loss = []
#         prompt_encoder.train()
#         for step, batch in enumerate(train_dataloader):
#             with torch.no_grad():
#                 token_embeds = text_encoder(**batch['prompt']).last_hidden_state
#             prompt_embeds = prompt_encoder(
#                 entity_ids=batch['entity'],
#                 token_embeds=token_embeds,
#                 output_entity=True,
#                 use_rec_prefix=True
#             )
#             batch['context']['prompt_embeds'] = prompt_embeds
#             batch['context']['entity_embeds'] = prompt_encoder.get_entity_embeds()

#             loss = model(**batch['context'], rec=True).rec_loss / args.gradient_accumulation_steps
#             accelerator.backward(loss)
#             train_loss.append(float(loss))

#             # optim step
#             if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
#                 if args.max_grad_norm is not None:
#                     accelerator.clip_grad_norm_(prompt_encoder.parameters(), args.max_grad_norm)
#                 optimizer.step()
#                 lr_scheduler.step()
#                 optimizer.zero_grad()

#                 progress_bar.update(1)
#                 completed_steps += 1
#                 if run:
#                     run.log({'loss': np.mean(train_loss) * args.gradient_accumulation_steps})

#             if completed_steps >= args.max_train_steps:
#                 break

#         # metric
#         train_loss = np.mean(train_loss) * args.gradient_accumulation_steps
#         logger.info(f'epoch {epoch} train loss {train_loss}')

#         del train_loss, batch

#         # valid
#         valid_loss = []
#         prompt_encoder.eval()
#         for batch in tqdm(valid_dataloader):
#             with torch.no_grad():
#                 token_embeds = text_encoder(**batch['prompt']).last_hidden_state
#                 prompt_embeds = prompt_encoder(
#                     entity_ids=batch['entity'],
#                     token_embeds=token_embeds,
#                     output_entity=True,
#                     use_rec_prefix=True
#                 )
#                 batch['context']['prompt_embeds'] = prompt_embeds
#                 batch['context']['entity_embeds'] = prompt_encoder.get_entity_embeds()

#                 outputs = model(**batch['context'], rec=True)
#                 valid_loss.append(float(outputs.rec_loss))
#                 logits = outputs.rec_logits[:, kg['item_ids']]
#                 ranks = torch.topk(logits, k=50, dim=-1).indices.tolist()
#                 ranks = [[kg['item_ids'][rank] for rank in batch_rank] for batch_rank in ranks]
#                 labels = batch['context']['rec_labels']
#                 evaluator.evaluate(ranks, labels)

#         # metric
#         report = accelerator.gather(evaluator.report())
#         for k, v in report.items():
#             report[k] = v.sum().item()

#         valid_report = {}
#         for k, v in report.items():
#             if k != 'count':
#                 valid_report[f'valid/{k}'] = v / report['count']
#         valid_report['valid/loss'] = np.mean(valid_loss)
#         valid_report['epoch'] = epoch
#         logger.info(f'{valid_report}')
#         if run:
#             run.log(valid_report)
#         evaluator.reset_metric()

#         if valid_report[f'valid/{metric}'] * mode > best_metric * mode:
#             prompt_encoder.save(best_metric_dir)
#             best_metric = valid_report[f'valid/{metric}']
#             logger.info(f'new best model with {metric}')

#         # test
#         test_loss = []
#         prompt_encoder.eval()
#         for batch in tqdm(test_dataloader):
#             with torch.no_grad():
#                 token_embeds = text_encoder(**batch['prompt']).last_hidden_state
#                 prompt_embeds = prompt_encoder(
#                     entity_ids=batch['entity'],
#                     token_embeds=token_embeds,
#                     output_entity=True,
#                     use_rec_prefix=True
#                 )
#                 batch['context']['prompt_embeds'] = prompt_embeds
#                 batch['context']['entity_embeds'] = prompt_encoder.get_entity_embeds()

#                 outputs = model(**batch['context'], rec=True)
#                 test_loss.append(float(outputs.rec_loss))
#                 logits = outputs.rec_logits[:, kg['item_ids']]
#                 ranks = torch.topk(logits, k=50, dim=-1).indices.tolist()
#                 ranks = [[kg['item_ids'][rank] for rank in batch_rank] for batch_rank in ranks]
#                 labels = batch['context']['rec_labels']
#                 evaluator.evaluate(ranks, labels)

#         # metric
#         report = accelerator.gather(evaluator.report())
#         for k, v in report.items():
#             report[k] = v.sum().item()

#         test_report = {}
#         for k, v in report.items():
#             if k != 'count':
#                 test_report[f'test/{k}'] = v / report['count']
#         test_report['test/loss'] = np.mean(test_loss)
#         test_report['epoch'] = epoch
#         logger.info(f'{test_report}')
#         if run:
#             run.log(test_report)
#         evaluator.reset_metric()

#     final_dir = os.path.join(args.output_dir, 'final')
#     prompt_encoder.save(final_dir)
#     logger.info(f'save final model')
