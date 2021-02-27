SQuAD2.0的JSON数据格式：

list of dict
     |--'title' -- 字符串标题
	 |--'paragraphs' -- list of dict
						     |--'qas' -- list of dict
										      |--'plausible_answers'-- list of dict（当is_impossible为True的时候，这个项有可能没有）
											                                |--'text' -- 字符串答案是啥
																			|--'answer_start' -- int整型开始的位置 
											  |--'question' -- 字符串问题
											  |--'id' -- 字符串id
											  |--'answers'-- list of dict
											           |--'text' -- 字符串答案是啥
													   |--'answer_start' -- int整型开始的位置
											  |--'is_impossible' -- 布尔值
							 |--'context' -- 字符串文本
							 


----------------------------------------------------------------------------------					 
统一将数据清洗成如下格式（JSON）：

list of dict
	 |--'context'
     |--'questions' -- list of dict
					 |--'question'
					 |--'is_impossible'
					 |--'id'
					 |--'answers' 		  	-- list of dict
											|--'text'
											|--'answer_start'
											|--'answer_end'
					 |--'plausible_answers'	-- list of dict
											|--'text'
											|--'answer_start'
											|--'answer_end'
	 

id遵循格式： 来自于哪个数据集 + id
'plausible_answers'：如果is_impossible为True，则提供一些似是而非的答案。有些数据集不提供可以留空list。
	 