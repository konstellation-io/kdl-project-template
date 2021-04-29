# KDL Project Template

## Project structure

```
├── lab
│   │
│   ├── analysis  <- Analyses of data, models etc. (typically notebooks)
│   │
│   ├── docs      <- High-level reports, executive summaries at each milestone (typically .md)
│   │
│   ├── lib       <- Importable functions shared between analysis notebooks and processes scripts
│   │                (including unit tests)
│   │
│   └── processes           <- Source code for reproducible workflow steps, for example:
│       ├── process_data   
│       │   ├── main.py      
│       │   ├── process_data.py  
|       │   └── test_process_data.py
|       ├── train_model
│       │   ├── main.py      
│       │   ├── train_model.py  
|       │   └── test_train_model.py
│       └── ...
│   
├── goals         <- Acceptance criteria (TBD)
│   
├── runtimes       <- Code for generating runtimes for deployment (.krt)
│   
├── .drone.yml     
│   
└── README.md
```
