
# coding: utf-8

# In[20]:


import os, json


# In[24]:


# collect all json files from 'json' directory
def getAllAnswers():
    json_paths = 'json/'
    json_files = [f for f in os.listdir(json_paths) if f.endswith('.json')]
    all_answers = {}

    # parse data from all json files
    for i, js in enumerate(json_files):

        key = js.replace(".json", "")
        all_answers[key] = []

        with open(os.path.join(json_paths, js)) as json_file:

            for line in json_file.readlines(): 
                all_answers[key].append(json.loads(line))
    # 'answers' is a dictionary of arrays, where each key represents a participant name and each 
    # entry indicates whether the corresponding answer was a truth or lie

    answers = {}          

    # 'honesty' is a dictionary of 3D arrays, where each key corresponds to a participant name and each 
    # entry follows this format: (question number, time step, signal type)
    # 
    # NOTE: signal type can be one of the following: lowGamma, highAlpha, lowBeta, 
    # highBeta, lowAlpha, delta, theta, midGamma

    honesty = {}   

    for participant, answers_dict in all_answers.items():

        if participant not in honesty:
            answers[participant] = []
            honesty[participant] = []

        for ans in answers_dict:
            honesty[participant].append(ans['truthful'])

            response_data = []

            for data_dict in ans['data']:

                data = []

                for signal, val in data_dict.items():

                    if signal != '_dataValueBytes':
                        data.append(val)

                response_data.append(data)

            answers[participant].append(response_data)
    return answers, honesty


# In[23]:



   


# In[ ]:




