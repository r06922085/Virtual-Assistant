def data_format(datatype):
    s = []
    d = []
    a = []
    
    verb = ['none','打開','關閉','調大','調小','打電話'];
    noun = ['機','wifi','藍芽','音量','銀幕','相機'];
    command = verb+noun;
    
    if datatype == 'command_dic':
        return command
        
    if datatype == 'task_name':
        task_name = [
        'turn_off_device',
        'display_lighter',
        'display_darker',
        'volumn_louder',
        'volumn_lower',
        'taking_pic',
        'turn_on_wifi',
        'turn_off_wifi',
        'turn_on_bluetooth',
        'turn_off_bluetooth',
        'name',
        ]
        
        return task_name
        
    
    if datatype == 'turn_off_device':
        
        s.append(['關機'])
        
        d.append("關閉")
        d.append("機")
        
        a.append(['正在關機'])
        
    if datatype == 'display_lighter':
        
        s.append(['銀幕', ''])
        s.append(['亮', '太暗'])
        
        d.append("調大")
        d.append("銀幕")
        
        a.append(['正在調亮銀幕'])
        
    if datatype == 'display_darker':
        
        s.append(['銀幕', ''])
        s.append(['暗', '太亮'])
        
        d.append("調小")
        d.append("銀幕")
        
        a.append(['正在調暗銀幕'])
        
    if datatype == 'volumn_louder':
        
        s.append(['音量', '聲音', ''])
        s.append(['大聲', '太小聲'])
        
        d.append("調大")
        d.append("音量")
        
        a.append(['正在調大音量'])
        
    if datatype == 'volumn_lower':
        
        s.append(['音量', '聲音', ''])
        s.append(['小聲', '太大聲'])
        
        d.append("調小")
        d.append("音量")
    
        a.append(['正在調小音量'])
        
    if datatype == 'taking_pic':
        
        s.append(['打開, 開啟', ''])
        s.append(['相機', '拍照', '照相'])
        
        d.append("打開")
        d.append("相機")
        
        a.append(['正在打開相機'])
        
    if datatype == 'turn_on_wifi':
        
        s.append(['打開', '開啟', ''])
        s.append(['wifi', '網路'])
        
        d.append("打開")
        d.append("wifi")
        
        a.append(['正在打開wifi'])
        
    if datatype == 'turn_off_wifi':
        
        s.append(['關閉', '關掉'])
        s.append(['wifi', '網路'])

        d.append("關閉")
        d.append("wifi")
        
        a.append(['正在關閉wifi'])
        
    if datatype == 'turn_on_bluetooth':
        
        s.append(['打開', '開啟', ''])
        s.append(['藍芽'])
   
        d.append("打開")
        d.append("藍芽")           

        a.append(['正在打開藍芽'])        
        
    if datatype == 'turn_off_bluetooth':
        
        s.append(['關閉', '關掉'])
        s.append(['藍芽'])
        
        d.append("關閉")
        d.append("藍芽")
        
        a.append(['正在關閉藍芽'])
       

    if datatype == 'name':
        
        s.append(['你什麼名字?','你是誰?'])

        d.append("none")

        a.append(['我是佐臻虛擬助理'])        
    
    #將command文字轉成編碼
    d_ = [];
    for i in range(len(d)):
        d_.append(command.index(d[i]))
        
    return s, d_, a

import random
def random_phone_number():
    rand_digit_num = 10
    
    phone_number = '';

    for i in range(rand_digit_num):   
        phone_number +=  str(random.randint(0,9));
        
    return phone_number