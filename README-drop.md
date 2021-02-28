Drop dataset answer包含span类和非span类（count，日期，以及无法在paragraph中精确匹配的str）。其中非span类全部被归为了unanswerable question.

Train-drop 共有77408个问题，其中Answerable question 28272个，Unanswerable question 49136个。

Dev-drop 共有9536个问题，其中Answerable question 3389个，Unanswerable question 6147个。


训练集中每个问题只给了一个标准答案，而dev-set中还增加了validated_answers，但validated_answer有些是错的，目前清洗后的数据集过滤掉了validated_answer。问题与答案一对一。