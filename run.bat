#!/bin/zsh

python server_app.py &
python client_app.py US &
python client_app.py MEX &
python client_app.py CAN &
python client_app.py PR &
python client_app.py UNK &




