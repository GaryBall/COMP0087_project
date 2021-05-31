from torch.nn import CrossEntropyLoss, MSELoss


def train(args, train_dataset, model, tokenizer):
    """ Train the model: BART1+GRU"""
    """
    This train function is specially designed for training Casual Language Modeling. 
    The basic stucutre is taken from the HuggingFace transformer, but the details are 
    largely modified to fit our purpose, especially when processing each batch. 
    Please look through this function mainly focusing on the batch processing. 

    :param args: the parameters
    :param train_dataset: training dataset
    :param model: Model for training
    :param tokenizer: tokenizer model
    :return: loss, global step and a example batch. 
    """

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    try:
        from apex import amp
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    model, optimizer = amp.initialize(model, optimizer, opt_level="O2")

    # Train!
    global_step = 0
    epoch = 0
    tr_loss, logging_loss = 0.0, 0.0

    # hyparameters for modified loss, check this in our chapter 3.4 of the report
    alpha = 1.2
    lmbda = 0.7
    model.zero_grad()

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=True)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)

        # in the second round of training, the CDM is fixed, and the lambda increases
        # to heavily penalize the incoherent sentence.
        if epoch == 1:
            lmbda = 1.5
            fix_layers = [model.gru, model.gru_head, model.linear_activation]
            # mybart.units_logit, mybart.units_activation]
            for layer in fix_layers:
                for parameter in layer.parameters():
                    parameter.requires_grad = False

        epoch += 1
        for step, batch in enumerate(epoch_iterator):

            batch_drop = 0
            token_list = tokenizer.convert_tokens_to_ids(["<INSTR_START>",
                                                          "<INGR_START>",
                                                          "<NEXT_INSTR>",
                                                          "<RECIPE_END>"])
            ins_start_id = token_list[0]
            ing_start_id = token_list[1]
            ins_element_id = token_list[2]
            recipe_end_id = token_list[3]

            # randomly shuffle the sentence
            random_shuffle = torch.randint(2, (batch.size()[0], 1), device=args.device)

            tokens_list = []
            max_length = 0

            # process the batch for training
            for batch_no in range(len(batch)):
                # <INSTR_START> not found in the batch, this is an invalid sample
                if (batch[batch_no] == ins_start_id).nonzero().size()[0] == 0:
                    print("An error happens, break")
                    batch_drop = 1
                    break
                # shuffle the batch
                if random_shuffle[batch_no] == 1:
                    shuffle_batch, shuffle_result = shuffle_instruction(batch[batch_no],
                                                                        ins_start_id, ing_start_id)
                    random_shuffle[batch_no] = shuffle_result
                else:
                    shuffle_batch = batch[batch_no]

                # find the <NEXT_INSTR> in the batch
                ins_pos = ins_token_idf(batch[batch_no],
                                        ins_element_id, recipe_end_id)
                # if <NEXT_INSTR> not found, invalid
                if ins_pos == None:
                    batch_drop = 1
                    break

                ins_pos = ins_pos.reshape(-1, 1)

                # add the <NEXT_INSTR> to tokens_list for model input
                if ins_pos.size()[0] > max_length:
                    max_length = ins_pos.size()[0]
                ins_idx = torch.hstack((torch.full_like(ins_pos, batch_no), ins_pos))
                tokens_list.append(ins_idx)

            if batch_drop == 1:
                print("An error in this batch, break")
                torch.cuda.empty_cache()
                continue

            # for the purpose of text generation
            # the input and the label should be shifted by 1 token.
            inputs, labels = (batch[:, 0:-1], batch[:, 1:])
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            model.train()

            loss_CE = CrossEntropyLoss(reduction="sum")
            loss_BCE = BCELoss()

            # model feed-forward
            logits, consist_output = model(inputs, tokens_list)

            # loss adjustment
            bart_vocab_size = 50278
            loss1 = 0
            loss2 = 0

            for i in range(len(batch)):
                ing_end_index = (inputs[i] == ins_start_id).nonzero()[0]

                # modify the weights of different parts of the generation, as mentioned in chapter 3.4
                loss1 += alpha * loss_CE(logits[i, :ing_end_index].reshape(-1, bart_vocab_size),
                                         labels[i, :ing_end_index].reshape(-1))

                loss1 += loss_CE(logits[i, ing_end_index:].reshape(-1, bart_vocab_size),
                                 labels[i, ing_end_index:].reshape(-1))

            loss1 = loss1 / (len(batch) * 511)
            loss2 = loss_BCE(consist_output, random_shuffle.float())
            # control the weights of 2 training tasks.
            loss = loss1 + lmbda * loss2

            # output every 100 steps. This is designed to avoid browser crush
            if global_step % 100 == 0:
                logger.info(step)
                print("step", global_step)
                print("loss:", loss.item())

            # save every 10000 step.
            if global_step % 10000 == 0 and global_step != 0:
                model_checkpoint = model.bart
                model_class = BartForCausalLM
                tokenizer_class = BartTokenizer
                save_model(args, model_checkpoint, tokenizer, model_class, tokenizer_class)

            # automatic loss scale
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            tr_loss += loss.item()
            if (step) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

            if args.max_steps > 0 and global_step > 2:
                epoch_iterator.close()
                break
            del inputs, labels, logits, consist_output, loss1, loss2, loss
            torch.cuda.empty_cache()

    return global_step, tr_loss / global_step, batch



def train_bart2_gru(args, train_dataset, model, tokenizer):
    """ Train the model: BART2+GRU"""
    """
    This train function is specially designed for training Conditional 
    Generation Model of BART2 with a gru unit 

    :param args: the parameters
    :param train_dataset: training dataset
    :param model: Model for training
    :param tokenizer: tokenizer model
    :return: loss, global step and a example batch. 
    """

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    try:
        from apex import amp
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    model, optimizer = amp.initialize(model, optimizer, opt_level="O2")

    # Train!
    global_step = 0
    epoch = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=True)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)

        alpha = 1.3
        lmbda = 0.5
        if epoch == 1:
            lmbda = 1.5
            # print("start discriminative part")
            # second step: fix the discriminative part
            fix_layers = [model.gru, model.gru_head, model.linear_activation]
            # mybart.units_logit, mybart.units_activation]
            for layer in fix_layers:
                for parameter in layer.parameters():
                    parameter.requires_grad = False

        epoch += 1
        for step, batch in enumerate(epoch_iterator):

            batch_drop = 0
            token_list = tokenizer.convert_tokens_to_ids(["<INSTR_START>",
                                                          "<INGR_START>",
                                                          "<NEXT_INSTR>",
                                                          "<RECIPE_END>"])
            ins_start_id = token_list[0]
            ing_start_id = token_list[1]
            ins_element_id = token_list[2]
            recipe_end_id = token_list[3]

            random_shuffle = torch.randint(2, (batch.size()[0], 1), device=args.device)
            # print(random_shuffle)

            # shuffle_instruction(batch[1],ins_element_id,recipe_end_id).shape

            # ins_tokens_list = []
            tokens_list = []
            max_length = 0

            encoder_batch = torch.empty(0, 48)
            decoder_batch = torch.empty(0, 512)
            for batch_no in range(len(batch)):

                if (batch[batch_no] == ins_start_id).nonzero().size()[0] == 0:
                    print("An error happens, break")
                    batch_drop = 1
                    break

                if random_shuffle[batch_no] == 1:
                    shuffle_batch = shuffle_instruction(batch[batch_no],
                                                        ins_start_id, ing_start_id)
                else:
                    shuffle_batch = batch[batch_no]

                # split the encoder & decoder input
                encoder_input, decoder_input = split_ing_dirs(shuffle_batch)

                # encoder & decoder input list
                encoder_batch = torch.cat((encoder_batch, encoder_input.unsqueeze(0)))
                decoder_batch = torch.cat((decoder_batch, decoder_input.unsqueeze(0)))

                ins_pos = ins_token_idf(batch[batch_no],
                                        ins_element_id, recipe_end_id)
                if ins_pos == None:
                    batch_drop = 1
                    break

                # reshape for stacking
                ins_pos = ins_pos.reshape(-1, 1)

                if ins_pos.size()[0] > max_length:
                    max_length = ins_pos.size()[0]
                ins_idx = torch.hstack((torch.full_like(ins_pos, batch_no), ins_pos))
                tokens_list.append(ins_idx)

            if batch_drop == 1:
                print("An error in this batch, break")
                torch.cuda.empty_cache()
                continue

            # prepare the batch for GPU training
            encoder_batch = encoder_batch.long().to(args.device)
            decoder_batch = decoder_batch.long().to(args.device)
            inputs, labels = (decoder_batch[:, 0:-1], decoder_batch[:, 1:])
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            model.train()

            loss_CE = CrossEntropyLoss(reduction="sum")
            loss_BCE = BCELoss()

            # model feed-forward
            logits, consist_output = model(encoder_batch, inputs, tokens_list)

            # loss adjustment
            bart_vocab_size = 50278
            loss1 = 0
            loss2 = 0

            # hyperparameter in multi-task training

            # print(logits.shape, labels.shape)

            for i in range(len(batch)):
                ing_end_index = (inputs[i] == ins_start_id).nonzero()[0]
                # print(logits[i,:ing_end_index].reshape(-1, bart_vocab_size).shape)
                # print(labels[i,:ing_end_index].reshape(-1).shape)
                loss1 += alpha * loss_CE(logits[i, :ing_end_index].reshape(-1, bart_vocab_size),
                                         labels[i, :ing_end_index].reshape(-1))
                loss1 += loss_CE(logits[i, ing_end_index:].reshape(-1, bart_vocab_size),
                                 labels[i, ing_end_index:].reshape(-1))

            loss1 = loss1 / (len(batch) * 511)

            loss2 = loss_BCE(consist_output, random_shuffle.float())
            loss = loss1 + lmbda * loss2

            if global_step % 100 == 0:
                logger.info(step)
                print("step", global_step)
                print("loss:", loss.item())

            if global_step % 10000 == 0 and global_step != 0:
                model_checkpoint = model.bart
                model_class = BartForConditionalGeneration
                tokenizer_class = BartTokenizer
                save_model(args, model_checkpoint, tokenizer, model_class, tokenizer_class)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                # print(scaled_loss)
                scaled_loss.backward()

            tr_loss += loss.item()
            if (step) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

            if args.max_steps > 0 and global_step > 2:
                epoch_iterator.close()
                break
            del inputs, labels, logits, consist_output, loss1, loss2, loss, encoder_batch, decoder_batch
            torch.cuda.empty_cache()
            # break

    return global_step, tr_loss / global_step, batch



def train_bart2(args, train_dataset, model, tokenizer):
    """ Train the model: vanilla BART2"""
    """
    This train function is specially designed for training Conditional 
    Generation Model of vanilla BART2 

    :param args: the parameters
    :param train_dataset: training dataset
    :param model: Model for training
    :param tokenizer: tokenizer model
    :return: loss, global step and a example batch. 
    """

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    try:
        from apex import amp
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    model, optimizer = amp.initialize(model, optimizer, opt_level="O2")

    # Train!
    global_step = 0
    epoch = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=True)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)

        epoch += 1
        for step, batch in enumerate(epoch_iterator):

            batch_drop = 0
            token_list = tokenizer.convert_tokens_to_ids(["<INSTR_START>",
                                                          "<INGR_START>",
                                                          "<NEXT_INSTR>",
                                                          "<RECIPE_END>"])
            ins_start_id = token_list[0]
            ing_start_id = token_list[1]
            ins_element_id = token_list[2]
            recipe_end_id = token_list[3]

            # random_shuffle = torch.randint(2, (batch.size()[0], 1),device = args.device)
            random_shuffle = torch.randint(1, (batch.size()[0], 1), device=args.device)

            tokens_list = []
            max_length = 0

            encoder_batch = torch.empty(0, 48)
            decoder_batch = torch.empty(0, 512)
            for batch_no in range(len(batch)):
                if (batch[batch_no] == ins_start_id).nonzero().size()[0] == 0:
                    print(tokenizer.decode(batch[batch_no]))
                    print("An error happens, break")
                    batch_drop = 1
                    break

                # randomly shuffle the batch
                if random_shuffle[batch_no] == 1:
                    shuffle_batch = shuffle_instruction(batch[batch_no],
                                                        ins_start_id, ing_start_id)
                else:
                    shuffle_batch = batch[batch_no]
                # split the encoder & decoder input
                encoder_input, decoder_input = split_ing_dirs(shuffle_batch)

                # encoder & decoder input list
                encoder_batch = torch.cat((encoder_batch, encoder_input.unsqueeze(0)))
                decoder_batch = torch.cat((decoder_batch, decoder_input.unsqueeze(0)))

            if batch_drop == 1:
                print("An error in this batch, break")
                torch.cuda.empty_cache()
                continue

            # prepare the batch for GPU training
            encoder_batch = encoder_batch.long().to(args.device)
            decoder_batch = decoder_batch.long().to(args.device)
            inputs, labels = (decoder_batch[:, 0:-1], decoder_batch[:, 1:])
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            model.train()

            loss_CE = CrossEntropyLoss()
            loss_MSE = MSELoss()

            # model feed-forward
            # logits, consist_output = model(encoder_batch, inputs, tokens_list)
            output = model(input_ids=encoder_batch, decoder_input_ids=inputs)
            logits = output["logits"]
            # loss adjustment
            bart_vocab_size = 50278
            loss = loss_CE(logits.reshape(-1, bart_vocab_size), labels.reshape(-1))
            # loss2 = loss_MSE(consist_output, random_shuffle.float())
            # loss = loss1 + loss2
            # print("loss comp: ", loss, outputs['loss'])
            if global_step % 100 == 0:
                logger.info(step)
                print("step", global_step)
                print("loss:", loss.item())

            if global_step % 10000 == 0 and global_step != 0:
                model_checkpoint = model
                model_class = BartForConditionalGeneration
                tokenizer_class = BartTokenizer
                save_model(args, model_checkpoint, tokenizer, model_class, tokenizer_class)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            # print("cp 4")
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                # print(scaled_loss)
                scaled_loss.backward()
            # print("cp 5")
            tr_loss += loss.item()
            if (step) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

            if args.max_steps > 0 and global_step > 2:
                epoch_iterator.close()
                break
            # del inputs, labels,logits,consist_output , loss1,loss2, loss
            del inputs, labels, logits, loss, encoder_batch, decoder_batch, output
            torch.cuda.empty_cache()
            # break

    return global_step, tr_loss / global_step, (batch, encoder_batch, decoder_batch)