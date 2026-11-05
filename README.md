# HOPE-Foundation-Model

Finished:  
- Data downloader/pipeline
- Embed layer
- Classification head
- CMS model *being verified*

Todo:  
- Implement titans model
- Wrap the HOPE Foundation Model (Titans + CMS) together
- Implement Masking pretraining loss function
- Create a throwaway linear layer to shrink output CLS token from 128 to 8 for mask pretraining loss function
- Implement Contrastive Learning loss function (needs to be planned out still)
- Unit test data downloader
- Unit test embed
- Unit test Classification (I think)
