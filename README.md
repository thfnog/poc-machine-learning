# Preparar o ambiente
    $ virtualenv env
    $ source env/bin/activate
    $ pip install -r requirements.txt
    $ pip freeze > requirements.txt --> for new imports

# Para criar o modelo
    $ python train_model.py
    # Irá gerar o arquivo model.pickle com o modelo salvo

# Rodar a api
    $ python api.py

# Chamada de recurso, onde o '1' é o userId
    curl --location --request GET 'http://127.0.0.1:5000/attractions/1'



