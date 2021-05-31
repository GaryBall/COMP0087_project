# model saving
def save_model(args,model,tokenizer,model_class,tokenizer_class):
  # This function is taken from the huggingface transformer

  # Create output directory if neede
  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

  logger.info("Saving model checkpoint to %s", args.output_dir)
  # Save a trained model, configuration and tokenizer using `save_pretrained()`.
  # They can then be reloaded using `from_pretrained()`
  model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
  model_to_save.save_pretrained(args.output_dir)
  tokenizer.save_pretrained(args.output_dir)

  # Good practice: save your training arguments together with the trained model
  torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

  # Load a trained model and vocabulary that you have fine-tuned
  model = model_class.from_pretrained(args.output_dir)
  tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
  model.to(args.device)


# model initialization
def model_init_vanillaCG(args,logger, model_class, tokenizer):
  if args.eval_data_file is None and args.do_eval:
    raise ValueError("Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
                         "or remove the --do_eval argument.")
  if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
    raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
  device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
  args.n_gpu = torch.cuda.device_count()
  args.device = device

  # Setup logging
  logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)

  # to initialize BART2+GRU, BART1+GRU, or BART1, change here
  # model = bartWithGRU(tokenizer)
  model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
  model.resize_token_embeddings(len(tokenizer))

  if args.block_size <= 0:
    args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
  args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
  model.to(args.device)

  logger.info("Training/evaluation parameters %s", args)
  return model, logger



def model_init_bartMLP(args,logger, model_class, tokenizer):
  if args.eval_data_file is None and args.do_eval:
    raise ValueError("Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
                         "or remove the --do_eval argument.")
  if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
    raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
  device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
  args.n_gpu = torch.cuda.device_count()
  args.device = device

  # Setup logging
  logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)

  model = bartForBinaryClassifier(tokenizer)

  if args.block_size <= 0:
    args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
  args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
  model.to(args.device)

  logger.info("Training/evaluation parameters %s", args)
  return model, logger


# model evaluation
def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = batch.to(args.device)

        with torch.no_grad():
            outputs = model(batch, labels=batch)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {
        "perplexity": perplexity
    }

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def init_tokenizer():
  tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
  # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
  special_tokens = {
    "additional_special_tokens": [
        "<TITLE_START>",
        "<TITLE_END>",
        "<INSTR_START>",
        "<NEXT_INSTR>",
        "<INSTR_END>",
        "<INGR_START>",
        "<NEXT_INGR>",
        "<INGR_END>",
        "<RECIPE_START>",
        "<RECIPE_END>",
        "<INPUT_START>",
        "<INPUT_END>",
        "<NEXT_INPUT>"
      ]
  }
  tokenizer.add_special_tokens(special_tokens)
  return tokenizer