import os
class Config():
    def __init__(self, 
                model_dir : str = "./models_forclip", 
                dataset_dir : str = "",
                images_dir : str = "",
                captions_dir : str = "",
                class_names : list[str] = "",
                image_size : tuple[int, int] = (224,224),
                sequence_lenght : int = 32,
                vocab_size : int = 10000,
                num_heads : int = 4,
                ff_dim : int = 256,
                num_layers : int = 2,
                embed_dim : int = 128
                ):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.dataset_dir = dataset_dir 
        self.images_dir = os.path.join(dataset_dir, images_dir)
        self.captions_dir = os.path.join(dataset_dir,captions_dir)
        self.class_names = class_names
        self.image_size = image_size
        self.image_shape = image_size + (3,)
        self.sequence_lenght = sequence_lenght
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.embed_dim = embed_dim