const database = 'pizzeria';

use(database);

db.createCollection('clients');
db.createCollection('orders');
db.createCollection('products');
db.createCollection('categories');
db.createCollection('stores');
db.createCollection('employees');

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
    "price": 200.20,
    "delivery_employee_id": 1,
    "delivery": "2024-03-10 10:15:20",
    "orders": [
        {
            "product_id": 1,
            "quantity": 2,
            "price": 14.99
        },
        {
            "product_id": 5,
            "quantity": 1,
            "price": 24.99
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

db.stores.insert({
    "_id": 1,
    "address": "Calle Provenza",
    "postal_code": "08002",
    "location": "Cataluña",
    "province": "Barcelona",
})

db.employees.insert({
    "_id": 1,
    "name": "Juan",
    "surename": "Perez Gomez",
    "nif": "X12345678",
    "phone_number": "+987654321",
    "role": "delivery",
    "store_id": 1
})
