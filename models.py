import pandas as pd
from surprisal import AutoHuggingFaceModel, OpenAIModel
import openai

class LM():
    def __init__(self, model_name, model_type, openai_key=None, openai_org=None):
        self.model_name = model_name
        self.model_type = model_type
        if model_type == "hf":
            # NOTE: "model_class" needs to be "gpt" for causal LMs,
            # but the tokenizer and model will still be loaded according to `model_name`
            # See https://github.com/aalok-sathe/surprisal/issues/19
            self.m = AutoHuggingFaceModel.from_pretrained(
                model_name, model_class="gpt"
            )
            try:
                self.m.to("cuda")
            except:
                self.m.device = "cpu"
        else:
            self.m = OpenAIModel(
                model_id=model_name,
                openai_api_key=openai_key, 
                openai_org=openai_org
            )
            openai.api_key = openai_key
            openai.organization = openai_org

    def _get_token_surprisals(self, text):
        [token_surprisals] = self.m.surprise(text)
        return token_surprisals
    
    def sentence_surprisal(self, sentence):
        token_surprisals = self._get_token_surprisals(sentence)
        sum_surprisal = sum(token_surprisals.surprisals)
        token_surprisal_df = pd.DataFrame({
            "token": token_surprisals.tokens,
            "surprisal": token_surprisals.surprisals,
            "token_id": list(range(len(token_surprisals.tokens)))
        })
        return sum_surprisal, token_surprisal_df
    
    def generate(self, input):
        if self.model_type == "hf":
            inputs = self.m.tokenizer(input, return_tensors="pt").to(self.device)
            generate_ids = self.m.model.generate(**inputs, max_new_tokens=5)
            output = self.m.tokenizer.batch_decode(
                generate_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
            )[0]
        else:
            request_kws = dict(
                engine=self.model_name,
                prompt=input,
                temperature=0.5,
                logprobs=1,
                echo=False
            )
            response = openai.Completion.create(**request_kws)["choices"][0]
            output = response["text"].strip()
        return output