## To train and test the domain classification model:
python domain_classification.py --train

python domain_classification.py --test

## To train and test the slot filling model:
python slot_filling.py --train

python slot_filling.py --test

## To create the domain classification/slot filling data:
python create_all_data --datatype create_all

## The things that virtual assistant can do:
* 打開app
* 計算機
* 搜尋
* 查時間
* 提醒(行事曆)
* 定鬧鐘
* 天氣
* 打電話
* 關機
* 銀幕調亮/暗
* 音量調大/小
* 拍照
* 打開/關閉 wifi
* 打開/關閉 bluetooth

## The basic idea of the virtual assistant:
To achieve the goal, we need to let virtaul assistant understanding our language. It mainly depart to two part:
* domain classification
* slot filling

First part is to determine which domain of goal. For example, what's the weather is for "weather" 
and turn on the wifi is for "turn_on_wifi"

And the second part is slot filling, it is the further step after knowing the domain. 
For example, when we know that it's the "weather" doamin, we need to know more information like 'where' 、 'when'.
This is the job that be done on this step
