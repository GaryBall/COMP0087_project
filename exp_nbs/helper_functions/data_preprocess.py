from collections import OrderedDict

# data preprocessing
# this function is taken from the recipeNLG codes
def data_drop(df):


  # calculate length of the removal items
  # the conditions are applied to title, ingredients and directions
  remove1 = df.loc[df.title.map(lambda x: len(x)<4 )]
  remove2 = df.loc[df.ingredients.map(lambda x: len(x)<2)]
  remove3 = df.loc[df.directions.map(lambda x: len(x) < 2 or len(''.join(x)) < 30)]
  remove4 = df.loc[df.directions.map(lambda x: re.search('(step|mix all)', ''.join(str(x)), re.IGNORECASE)!=None)]
  len(remove3)+len(remove2)+len(remove1)+len(remove4)
  # remove the items
  df.drop(df[df.title.map(lambda x: len(x)<4)].index, inplace=True)
  df.drop(df[df.ingredients.map(lambda x: len(x)<2)].index, inplace=True)
  df.drop(df[df.directions.map(lambda x: len(x) < 2 or len(''.join(x)) < 30)].index, inplace=True)
  df.drop(df[df.directions.map(lambda x: re.search('(step|mix all)', ''.join(str(x)), re.IGNORECASE)!=None)].index, inplace=True)

  return df



def df_to_plaintext_file(input_df, output_file, train=True):
    """
    prepare data in txt type for further use
    :param input_df:
    :param output_file:
    :param train:
    :return:
    """
    print("Writing to", output_file)

    with open(output_file, 'w') as f:
        for index, row in input_df.iterrows():
            if index % 100000 == 0:
                print(index)
                print(res)
            if type(row.NER) != str:
                continue
            title = row.title
            directions = json.loads(row.directions)
            if len(directions) <= 1 and train:
                continue
            # print(len(directions))
            ingredients = json.loads(row.ingredients)
            ner = json.loads(row.NER)
            # print(ner)
            res = "<RECIPE_START> <INPUT_START> " + " <NEXT_INPUT> ".join(ner) + " <INPUT_END> <INGR_START> " + \
                  " <NEXT_INGR> ".join(ingredients) + " <INGR_END> <INSTR_START> " + \
                  " <NEXT_INSTR> ".join(
                      directions) + " <INSTR_END> <TITLE_START> " + title + " <TITLE_END> <RECIPE_END>"
            # print(res)
            f.write("{}\n".format(res))


def datapre_txt2h5_CLM(tokenizer,train_size, test_size):
    """
    prepare data from txt to h5 file for casual language modelling
    :param tokenizer: pretrained tokenizer type
    :param train_size: training size
    :param test_size: testing size
    :return: None
    """
    end_token_id = tokenizer.convert_tokens_to_ids(["<RECIPE_END>"])[0]
    ing_token_id = tokenizer.convert_tokens_to_ids(["<INPUT_END>"])[0]
    directions_size = 512
    hf = h5py.File("unsupervised.h5", "w")
    for filename in ["test", "train"]:
        out_np = []
        data = open("unsupervised_" + filename + ".txt", "r")
        num = 0
        rows = 0
        for line in data:
            num += 1
            if num % 10000 == 0:
                print("Read " + str(num) + " Written: " + str(rows))
            text_tokens = tokenizer(line)['input_ids']
            if ing_token_id in text_tokens:
                pass
            else:
                continue

            # error in one recipe
            # 50273 is the token for <RECIPE_END>
            if len(text_tokens) > directions_size or (50273 not in text_tokens):
                continue

            text_tokens_ids = text_tokens

            while len(text_tokens_ids) < directions_size:
                text_tokens_ids.append(end_token_id)
            out_np.append(text_tokens_ids)
            rows += 1

            if rows == train_size and filename == 'train':
                print("training sample enough:", train_size)
                break
            if rows == test_size and filename == 'test':
                print("testing sample enough", test_size)
                break
        out_mat = np.matrix(out_np)
        print(out_mat.shape)
        hf.create_dataset(filename, data=out_mat)
    hf.close()






def datapre_txt2h5_CG(tokenizer, train_size, test_size):
    """
    prepare data from txt to h5 file for conditional generation
    :param tokenizer: pretrained tokenizer type
    :param train_size: training size
    :param test_size: testing size
    :return: None
    """

    end_token_id = tokenizer.convert_tokens_to_ids(["<RECIPE_END>"])[0]
    ing_token_id = tokenizer.convert_tokens_to_ids(["<INPUT_END>"])[0]

    next_input_token_id = tokenizer.convert_tokens_to_ids(["<NEXT_INPUT>"])[0]

    directions_size = 512
    hf = h5py.File("unsupervised.h5", "w")
    for filename in ["test", "train"]:
        out_np = []
        data = open("control_tokens_" + filename + ".txt", "r")
        num = 0
        rows = 0
        for line in data:
            num += 1
            if num % 10000 == 0:
                print("Process " + str(num) + "; Valid: " + str(rows))
            text_tokens = tokenizer(line)['input_ids']

            # generate the encoder input
            if ing_token_id in text_tokens:
                ing_idx = text_tokens.index(ing_token_id)
                temp_list = list(OrderedDict.fromkeys(text_tokens[3:ing_idx]))

                # if <NEXT_INPUT> not in list, do not need to process
                if next_input_token_id in temp_list:
                    temp_list.remove(next_input_token_id)
                else:
                    continue
                text_tokens = temp_list + text_tokens
            else:
                continue

            # error in one recipe
            if len(text_tokens) > directions_size or (50273 not in text_tokens):
                continue

            # text_tokens_ids = tokenizer.convert_tokens_to_ids(text_tokens)
            text_tokens_ids = text_tokens

            # append <RECIPDE_END> token to the end
            while len(text_tokens_ids) < directions_size:
                text_tokens_ids.append(end_token_id)
            out_np.append(text_tokens_ids)
            rows += 1

            if rows == train_size and filename == 'train':
                print("training sample enough:", train_size)
                break
            if rows == test_size and filename == 'test':
                print("testing sample enough", test_size)
                break
        out_mat = np.matrix(out_np)
        print(out_mat.shape)
        hf.create_dataset(filename, data=out_mat)
    hf.close()


# data loading
class TextDataset(Dataset):

    def __init__(self, tokenizer, file_path='train', block_size=512):
        cached_features_file = "unsupervised.h5"

        logger.info("Loading features from cached file %s", cached_features_file)
        with h5py.File(cached_features_file, 'r') as f:
            if file_path=='test':
                self.examples = f[file_path][:] #this is a dev set, 10% of a test set
            else:
                self.examples = f[file_path][:]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item])

def load_and_cache_examples(args, tokenizer, evaluate=False):
    dataset = TextDataset(tokenizer, file_path="test" if evaluate else "train", block_size=args.block_size)
    print(dataset[0:5])
    return dataset


train_size = 350000
test_size = 100

df_to_plaintext_file(train, 'unsupervised_train.txt')
df_to_plaintext_file(test, 'unsupervised_test.txt',train = False)

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
data_pre_for_encoder(tokenizer, train_size, test_size)