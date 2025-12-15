from pydantic import BaseModel
from datasets import Dataset, DatasetDict , load_dataset
from typing import Optional , Self, List

PREFIX="Price is $"
QUESTION="What is the price of the product, rounded to the nearest dollar?"

class Item(BaseModel):
    title: str
    category:str
    price:float
    description:Optional[str] = None
    weight: Optional[float] = None
    summary: Optional[str] = None
    prompt: Optional[str] = None
    completion: Optional[str] = None
    id: Optional[int] = None

    def prepare_prompt(self , text:str):
        return f"{QUESTION}\n\n{text}\n\n{PREFIX}{round(self.price)}.00"
    
    def test_prompt(self)->str:
        return self.prompt.split(PREFIX)[0] + PREFIX
    
    def __repr__(self):
        return f"<Title: {self.title} | Category:  {self.category} | Price {self.price}>"
    
    def count_tokens(self, tokenizer):
        """Count tokens in the summary"""
        return len(tokenizer.encode(self.summary, add_special_tokens=False))

    def make_prompts(self, tokenizer, max_tokens, do_round):
        """Make prompts and completions"""
        tokens = tokenizer.encode(self.summary, add_special_tokens=False)
        if len(tokens) > max_tokens:
            summary = tokenizer.decode(tokens[:max_tokens]).rstrip()
        else:
            summary = self.summary
        self.prompt = f"{QUESTION}\n\n{summary}\n\n{PREFIX}"
        self.completion = f"{round(self.price)}.00" if do_round else str(self.price)

    def count_prompt_tokens(self, tokenizer):
        """Count tokens in the prompt"""
        full = self.prompt + self.completion
        tokens = tokenizer.encode(full, add_special_tokens=False)
        return len(tokens)

    def to_datapoint(self) -> dict:
        return {"prompt": self.prompt, "completion": self.completion}

    @staticmethod
    def push_to_hub(dataset_name:str, train:List[Self], val:List[Self], test:List[Self]):
        DatasetDict(
            {
                "train":Dataset.from_list([item.model_dump() for item in train]),
                "val": Dataset.from_list([item.model_dump() for item in val]),
                "test": Dataset.from_list([item.model_dump() for item in test]),
            }
        ).push_to_hub(dataset_name)

    @classmethod
    def get_from_hub(cls , dataset_name:str)->tuple[List[Self], List[Self], List[Self]]:
        dataset = load_dataset(dataset_name)
        return (
            [cls.model_validate(train_row) for train_row in dataset['train']],
            [cls.model_validate(val_row) for val_row in dataset['validation']],
            [cls.model_validate(test_row) for test_row in dataset['test']],

        )



