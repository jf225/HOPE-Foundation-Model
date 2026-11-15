# HOPE-Foundation-Model

Finished:  
- Data downloader/pipeline
- Embed layer
- Classification head
- CMS model
- Implement titans model
- Wrap the HOPE Foundation Model (Titans + CMS) together
- Masking pretraining loss function
- Unit test CMS Module
- Contrastive Loss function for pretraining

Todo:  
- Unit test data downloader
- Unit test embed
- Unit test Classification (I think)
- Train and test baseline model for initial results
- CMS Adaptive Gating
- CMS MLP inversion
- Timescale Modification (Check the ucr data, most are probably too short for this to be viable)

Stretch Goals:
- Timeseries Forecasting
- Customized Loss function (to solve decreased accuracy on multiclass classification)
- Add statistical vector head to finetuning section (can be added without changing pretrained model so should be computationally cheap to test)
