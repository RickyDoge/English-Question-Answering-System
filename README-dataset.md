1. SQUAD数据集

train: Answerable 86821, Unanswerable 43498
dev: Answerable 5928, Unanswerable 5945

增强后：train: Unanswerable 132230, Answerable 132230

----------------------------------

2. DROP数据集

Drop dataset answer包含span类和非span类（count，日期，以及无法在paragraph中精确匹配的str）。其中非span类全部被归为了unanswerable question.

Train-drop 共有77408个问题，其中Answerable question 28272个，Unanswerable question 49136个。

Dev-drop 共有9536个问题，其中Answerable question 3389个，Unanswerable question 6147个。

训练集中每个问题只给了一个标准答案，而dev-set中还增加了validated_answers，但validated_answer有些是错的，目前清洗后的数据集过滤掉了validated_answer。问题与答案一对一。

处理后：
Dev: Answerable question count: 3389, Unanswerable question count: 6147

Train: Answerable question count: 28272, Unanswerable question count: 49136

增强后：Train: Answerable 45293, Unanswerable: 78428

----------------------------------

3. NewsQA数据集

train: Original Unanswerable: 32823, Answerable 81211

dev: Original Unanswerable: 1617, Answerable 3982

增强后：Train: Balanced Unanswerable: 122684, Answerable 122684

---------------------------------------------------

4. Wikihop数据集

train: Original Unanswerable: 166, Answerable 43572

dev: Original Unanswerable: 10, Answerable 5119

增强后：train: Unanswerable: 63484, Answerable 63484

-----------------------

5. Quoref数据集

train: Unanswerable: 0, Answerable 17412

dev: Unanswerable: 0, Answerable 2197

增强后：train: Unanswerable: 26585, Answerable 26585

---------------------------------

6. Medhop数据集

train: Unanswerable: 0, Answerable 1620

dev: Unanswerable: 0, Answerable 342

增强后：train: Unanswerable: 2620, Answerable 2620

