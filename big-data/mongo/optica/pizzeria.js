const database = 'pizzeria';

use(database);

db.createCollection('clients');
db.createCollection('orders');
db.createCollection('products');
db.createCollection('categories');

//dentificador únic, nom,
//cognoms, adreça, codi postal, localitat, província i número de telèfon
db.clients.insert({
    "_id": 1,
    "name": "Juan",
    "surename": "Perez Gomez",
    "address": "Calle Provenza 1, piso 3 puerta 2",
    "postalCode": "08010",
    "location": "Cataluña",
    "province": "Barcelona",
    "phone": "932121212",
})

db.orders.insert({
    "_id": 1,
    "client_id": 1,
    "shop_id": 1,
    "date": "2024-03-05 23:10:20",
    "type": "Domicili",
    "quantity": 5,
    "price": 200.20,
    "dealer": "Juan",
    "delivery": "2024-03-10 10:15:20",
    "orders": [
        {
            "product_id": 1,
            "quantity": 2,
        },
        {
            "product_id": 5,
            "quantity": 1,
        }
    ]
})


db.products.insert({
    "_id": 1,
    "name": "Pizza margarita",
    "description": "Pizza con queso y tomate",
    "category_id": 1,
    "image": "/images/pizza.png",
    "price": 45
})


db.categories.insert({
    "_id": 1,
    "name": "pizza",
})
