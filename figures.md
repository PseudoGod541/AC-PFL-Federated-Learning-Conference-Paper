graph TD
    %% Input
    Input["Input Sequence<br/>(N, 30, 24)"] --> Base

    %% Shared Base Network
    subgraph Base ["Shared Base Network (Globally Aggregated)"]
        L1["LSTM Layer 1<br/>(64 Units)"] --> BN1["Batch Normalization"]
        BN1 --> D1["Dropout (0.3)"]
        D1 --> L2["LSTM Layer 2<br/>(32 Units)"]
        L2 --> BN2["Batch Normalization"]
        BN2 --> D2["Dropout (0.3)"]
        D2 --> FC1["Fully Connected<br/>(32 Units, ReLU)"]
    end

    %% Private Prediction Head
    subgraph Head ["Private Prediction Head (Local to Client)"]
        FC1 --> FC2["Fully Connected<br/>(16 Units, ReLU)"]
        FC2 --> D3["Dropout (0.5)"]
        D3 --> Out["Output Layer<br/>(1 Unit, Linear)"]
    end

    %% Output
    Out --> RUL["Predicted RUL"]
