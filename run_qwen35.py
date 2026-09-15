#!/usr/bin/env python
"""Run Qwen3.5-4B GGUF with streamed weights on CPU or CUDA.

    python run_qwen35.py -p "Explain gravity simply."
    python run_qwen35.py --chat
    python run_qwen35.py --image photo.jpg -p "What is in this picture?"
    python run_qwen35.py --model models/Qwen3.5-4B          # safetensors checkpoint
    python run_qwen35.py --model models/qwen35/Qwen3.5-4B-Q4_K_M.gguf
    python run_qwen35.py --device cpu --ram-budget 2GB

Defaults to the unquantized BF16 model in models/qwen35. Download instructions
are in README.md. --model also accepts quantized GGUF files or a GGUF shard.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / 'models' / 'qwen35' / 'Qwen3.5-4B-BF16.gguf'


def is_checkpoint(path: Path) -> bool:
    """A HuggingFace checkpoint is a directory of shards, not a single file.

    Decided without importing torch, so --help stays instant.
    """
    return path.is_dir() and any(path.glob('*.safetensors'))


def describe_image(ns, args):
    """Stream one answer about an image."""
    for piece in ns.generate_vl(
        args.image, args.prompt, max_tokens=args.max_tokens,
        temperature=args.temperature, max_patches=args.max_patches,
        enable_thinking=args.thinking,
    ):
        print(piece, end='', flush=True)
    print('\n')
    print(f'  {ns.stats.summary()}', file=sys.stderr)


def reply(ns, messages, args):
    """Stream one reply and return its text for conversation history."""
    prompt = ns.tokenizer.apply_chat_template(messages, enable_thinking=args.thinking)
    ids = ns.tokenizer.encode(prompt)
    if len(ids) + args.max_tokens > ns.cfg.context_length:
        raise ValueError('Conversation exceeds the context limit. Use /reset or lower --max-tokens.')
    pieces = []
    for piece in ns.generate(ids, max_tokens=args.max_tokens,
                             temperature=args.temperature, top_p=args.top_p):
        pieces.append(piece)
        print(piece, end='', flush=True)
    print('\n')
    print(f'  {ns.stats.summary()}', file=sys.stderr)
    return ''.join(pieces)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run Qwen3.5-4B (BF16 or quantized GGUF).')
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL,
                        help='GGUF path; defaults to models/qwen35/Qwen3.5-4B-BF16.gguf')
    parser.add_argument('-p', '--prompt', default='Explain gravity simply.')
    parser.add_argument('--image', type=Path, default=None,
                        help='image to ask about; needs the matching mmproj file')
    parser.add_argument('--mmproj', type=Path, default=None,
                        help='vision tower GGUF; defaults to mmproj-BF16.gguf beside the model')
    parser.add_argument('--max-patches', type=int, default=1024,
                        help='cap on image tokens, so a large photo cannot flood the context')
    parser.add_argument('--chat', action='store_true', help='interactive chat with conversation history')
    parser.add_argument('-n', '--max-tokens', type=int, default=256)
    parser.add_argument('-t', '--temperature', type=float, default=0.0)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--thinking', action=argparse.BooleanOptionalAction, default=False,
                        help='enable thinking output; allow more tokens when enabled')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None,
                        help='default: CUDA when available, otherwise CPU')
    parser.add_argument('--mem-budget', default='512MB', help='streaming I/O buffer budget')
    parser.add_argument('--vram-budget', default=None, help='weight cache budget; default 6GB on CUDA, 0 on CPU')
    parser.add_argument('--ram-budget', default='0', help='host weight cache budget')
    parser.add_argument('--reserve-vram', default='1.5GB', help='VRAM reserved for activations and KV cache')
    parser.add_argument('--block-rows', type=int, default=1024)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args(argv)
    if args.max_tokens <= 0 or args.block_rows <= 0 or args.workers <= 0:
        parser.error('--max-tokens, --block-rows, and --workers must be positive')
    if args.max_patches <= 0:
        parser.error('--max-patches must be positive')
    if args.chat and args.image:
        parser.error('--image answers a single prompt; it cannot be combined with --chat')
    if args.temperature < 0 or not 0 < args.top_p <= 1:
        parser.error('--temperature must be nonnegative and --top-p must be in (0, 1]')
    args.model = args.model.expanduser()
    checkpoint = is_checkpoint(args.model)
    if args.image is not None:
        args.image = args.image.expanduser()
        if not args.image.is_file():
            parser.error(f'image not found: {args.image}')
        if args.mmproj is None and not checkpoint:
            args.mmproj = args.model.parent / 'mmproj-BF16.gguf'
        args.mmproj = args.mmproj.expanduser() if args.mmproj is not None else None
        if args.mmproj is not None and not args.mmproj.is_file():
            parser.error(
                f'vision tower not found: {args.mmproj}\n'
                'Download mmproj-BF16.gguf as shown in README.md, or pass --mmproj PATH.'
            )
    if not checkpoint and not args.model.is_file():
        parser.error(
            f'model not found: {args.model}\n'
            'Download the BF16 model using the command in README.md, or pass --model PATH.'
        )

    # Keep --help and missing-file checks lightweight.
    import torch
    from neurostream import NeuroStream
    from neurostream.format.gguf import GGUFFile
    from neurostream.format.safetensors import SafetensorsFile

    index = SafetensorsFile(args.model) if checkpoint else GGUFFile(args.model)
    if index.arch() != 'qwen35':
        parser.error('--model must point to a Qwen3.5 text GGUF or safetensors checkpoint')
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; use --device cpu')
    vram = args.vram_budget if args.vram_budget is not None else ('6GB' if device == 'cuda' else '0')
    started = time.perf_counter()
    with NeuroStream.load(
        args.model, device=device, mem_budget=args.mem_budget, vram_budget=vram,
        ram_budget=args.ram_budget, reserve_vram=args.reserve_vram,
        block_rows=args.block_rows, n_workers=args.workers, verbose=True,
    ) as ns:
        print(f'  Loaded {args.model.name} on {device} in {time.perf_counter() - started:.1f}s\n',
              file=sys.stderr)
        if args.image is not None:
            ns.attach_vision(args.mmproj)
            describe_image(ns, args)
            return 0
        if not args.chat:
            reply(ns, [{'role': 'user', 'content': args.prompt}], args)
            return 0

        print('Chat ready. /reset clears history; /quit exits.')
        messages = []
        while True:
            try:
                message = input('\nYou: ').strip()
            except EOFError:
                break
            if message.lower() in ('/quit', '/exit'):
                break
            if message.lower() == '/reset':
                messages.clear()
                print('History cleared.')
                continue
            if not message:
                continue
            pending = messages + [{'role': 'user', 'content': message}]
            print('Qwen: ', end='', flush=True)
            try:
                answer = reply(ns, pending, args)
            except ValueError as exc:
                print(f'\n{exc}', file=sys.stderr)
                continue
            messages = pending + [{'role': 'assistant', 'content': answer}]
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nStopped.', file=sys.stderr)
        raise SystemExit(130)
