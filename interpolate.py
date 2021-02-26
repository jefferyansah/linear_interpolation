import pandas as pd 
import numpy as np 


def predict_tonnes_loss(input_dict, target):
    
    input_list = list(input_dict.keys())
    
    if target in input_list:
        
        return input_dict[target]
    else:

        def get_keys(input_list, target):

            """
            Get the upper bound and lower 
                bound of a target in a given list  
            """
            final_list = input_list.copy()
            final_list.append(target)
            final_list = sorted(final_list)

            if final_list.index(target) == 0:
                return final_list[final_list.index(target)+ 1], final_list[final_list.index(target)+ 1]

            if final_list.index(target) == len(final_list) -1:
                return final_list[final_list.index(target)-1], final_list[final_list.index(target) -1]

            else:
                return final_list[final_list.index(target)-1], final_list[final_list.index(target)+ 1]

        lower_x, upper_x  = get_keys(input_list, target)

        def get_values(lower, upper, input_dict):
            return input_dict[lower], input_dict[upper]

        lower_y, upper_y = get_values(lower_x, upper_x, input_dict) 

        print([lower_x,lower_y], [upper_x, upper_y], target)

        def predict_new_y(lower_x,upper_x, lower_y, upper_y,target):
            grad = (upper_y - lower_y) / (upper_x - lower_x)
            pred = (target - upper_x) * grad + upper_y
            return pred
        pred = predict_new_y(lower_x,upper_x, lower_y, upper_y,target)
        
        return pred
    
input_dict = {10:500, 2:1000, 3:1500, 4:5000, 5:6000} 
predict_tonnes_loss(input_dict, 6)
