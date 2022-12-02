from collections import Counter
import numpy as np
import pandas as pd
import math
        

class LogKG:
    def __init__(self, train_case_log_df:pd.DataFrame, test_case_log_df:pd.DataFrame, idf_threshold:float, template_embedding) -> None:
        self.template_embedding = template_embedding
        self.train_case_log_df = train_case_log_df
        self.test_case_log_df = test_case_log_df
        self.idf_threshold = idf_threshold
        
    def get_train_idf(self):
        case_log_set_list = [list(set(df["EventId"].values)) for df in self.train_case_log_df.values()]
        case_all_template_occurrence = []
        for case_log_set in case_log_set_list:
            case_all_template_occurrence += case_log_set
        case_log_template_counter = dict(Counter(case_all_template_occurrence))
        self.template_list = list(case_log_template_counter.keys())
        template_idf = {}
        for template in case_log_template_counter:
            idf = math.log10(len(case_log_set_list) / case_log_template_counter[template])
            template_idf[template] = idf if idf > self.idf_threshold else 0.0
        self.template_idf = template_idf
        # print("IDF: ")
        # print(template_idf)
        
    def get_train_embedding(self, embedding_size=48):
        self.get_train_idf()
        case_embedding_dict = {}
        for key in self.train_case_log_df:
            embedding_array = np.zeros(len(self.template_idf), dtype=float)
            log_df = self.train_case_log_df[key]
            template_sequence = log_df["EventId"].values
            case_template_counter = dict(Counter(template_sequence))
            important_log_count = 0
            for template in case_template_counter:
                if self.template_idf[template] != 0:
                    important_log_count += case_template_counter[template]
                else:
                    case_template_counter[template] = 0
            case_embedding = np.zeros(embedding_size, dtype=np.float)
            if important_log_count == 0:
                case_embedding_dict[key] = case_embedding
                continue
            for template in case_template_counter:
                case_embedding += (case_template_counter[template] / important_log_count) * self.template_idf[template] * self.template_embedding[template]
            case_embedding_dict[key] = case_embedding
        self.train_embedding_dict = case_embedding_dict

    def get_test_embedding(self, embedding_size=48):
        case_embedding_dict = {}
        for key in self.test_case_log_df:
            embedding_array = np.zeros(len(self.template_idf), dtype=float)
            log_df = self.test_case_log_df[key]
            template_sequence = log_df["EventId"].values
            case_template_counter = dict(Counter(template_sequence))
            important_log_count = 0
            for template in case_template_counter:
                if template not in self.template_list:
                    continue
                if self.template_idf[template] != 0:
                    important_log_count += case_template_counter[template]
                else:
                    case_template_counter[template] = 0
            case_embedding = np.zeros(embedding_size, dtype=np.float)
            if important_log_count == 0:
                case_embedding_dict[key] = case_embedding
                continue
            for template in case_template_counter:
                if template not in self.template_list:
                    continue
                case_embedding += (case_template_counter[template] / important_log_count) * self.template_idf[template] * self.template_embedding[template]
            case_embedding_dict[key] = case_embedding
        self.test_embedding_dict = case_embedding_dict
