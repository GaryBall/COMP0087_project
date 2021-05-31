import random

def shuffle_instruction(this_batch,ins_element_id,recipe_end_id):
  """
  shuffle function to produce incoherent samples
  :param this_batch: The batch to be processed
  :param ins_element_id: The token of <NEXT_INSTR>
  :param recipe_end_id: The token of <RECIPE_END>
  :return: shuffled batch and feedback signal, as discussed in chapter 3.3 of report
  """

  # find the end of recipe
  end_index = (this_batch == recipe_end_id).nonzero()[0]

  # find the position of <NEXT_INSTR> tokens.
  ing_index = (this_batch == ins_element_id).nonzero()
  ing_index = ing_index[ing_index<end_index]

  # If more than 2 steps of directions found, shuffle this sample.
  if len(ing_index) > 1:
    # convert sentence to list, then
    # use random.shuffle function to shuffle the list
    split_result = torch.tensor_split(this_batch, ing_index.squeeze())
    split_list = list(split_result[1:-1])
    random.shuffle(split_list)
    shuffle_batch = torch.cat((split_result[0],torch.cat(split_list),split_result[-1]))
    return shuffle_batch, 1
  else:
    return this_batch, 0


def ins_token_idf(this_batch,ins_element_id,recipe_end_id):
  """
  identify the <NEXT_INSTR> token in a batch, this is necessary for CDM module
  :param this_batch: The batch to be processed
  :param ins_element_id: The token of <NEXT_INSTR>
  :param recipe_end_id: The token of <RECIPE_END>
  :return: the identified position of <NEXT_INSTR>
  """
  if (this_batch == recipe_end_id).nonzero().size()[0] == 0:
    return None
  end_index = (this_batch == recipe_end_id).nonzero()[0]
  ing_index = (this_batch == ins_element_id).nonzero()
  ing_index = ing_index[ing_index<end_index]
  if len(ing_index) > 0:
    return ing_index
  else:
    return None



def split_ing_dirs(this_batch):
  """
  Only used in Conditional generation; for Casual LM it is not.
  Split ingredients and directions, to respectively prepare
  the input of encoder and decoder sides. (for conditional generation)
  :param this_batch: the input batch
  :return: the processed ingredients and directions
  """
  token_list = tokenizer.convert_tokens_to_ids(["<INSTR_START>",
                                                "<INGR_START>",
                                                "<NEXT_INSTR>",
                                                "<RECIPE_END>",
                                                "<INPUT_END>",
                                                "<RECIPE_START>"])
  ins_start_id = token_list[0]
  ing_start_id = token_list[1]
  ins_element_id = token_list[2]
  recipe_end_id = token_list[3]
  input_end_id = token_list[4]
  recipe_start_id = token_list[5]

  sample = this_batch.clone()

  # ingredients size and instructions (directions) size
  ing_size = 48
  ins_size = 512

  # identify the ingredients and instructions in a batch
  ins_index = (sample == recipe_start_id).nonzero()[0] - 1

  # construct a tensor to store the ingredients
  ingredients = torch.zeros(ing_size)
  ingredients = torch.full_like(ingredients, input_end_id)
  if ins_index > ing_size:
    print("size overflow")
    ingredients[0:ing_size] = this_batch[0:ing_size]
  else:
    ingredients[0:ins_index] = this_batch[0:ins_index]
  sample[0:ins_index - 1] = 0
  nz = sample.nonzero().squeeze()
  # move the non-zero elements of instructions vector to left
  directions = torch.zeros(sample.numel() - nz.numel())
  directions = torch.full_like(directions, recipe_end_id)
  # directions = this_batch[-1]
  directions = torch.cat((sample[nz], directions))
  directions[0] = 0
  del sample
  return ingredients, directions

