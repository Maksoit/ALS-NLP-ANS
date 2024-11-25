Инструкция по запуску:
1. Открыть терминал Bash в директории проекта
2. Написать команды: 
    python -m venv venv
    source venv/Scripts/activate
    pip install -r requirements.txt
3. Запустить исходный код в файле main.py

Исправления:
1. Синтаксические:
1.1 assert в LanguageModel -> Forward -- исправлено неверное использование assert с комментарием при невыполнении проверки
1.2 assert в Attention -> __init__ -- добавлен поясняющий комментарий при невыполнении проверки
1.3 списки vocab и batch -- строки окружены кавычками
2. логические:
2.1 в LanguageModel -> __init__ -- nn.Linear заменен на nn.Embedding, так как nn.Embedding преобразует индексы токенов в численные вектора учитывая семантику текста
2.2 В TransformerBlock -> __init__ -- в self.mlp добавлена ReLU для корректной работы
3. Концептуальные:
3.1 В Attention -> forward -- был добавлен F.softmax для нормализации весов
3.2 В TransformerBlock -> __init__ -- nn.BatchNorm1d заменен на nn.LayerNorm, так как он лучше работает с последовательностями
3.3 В TransformerBlock -> forward -- добавил сохранение остаточных связей, так как это важная часть архитектуры трансформера
