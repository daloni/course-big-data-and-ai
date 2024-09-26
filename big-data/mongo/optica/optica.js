const database = 'optica';

use(database);

db.createCollection('providers');
db.createCollection('glasses');
db.createCollection('clients');

db.providers.insert({
    "_id": 1,
    "name": "Gafas SL",
    "phone": "931111111",
    "fax": "931111111",
    "NIF": "45.612.312J",
    "address": {
        "street": "Calle provenza",
        "number": 123,
        "floor": 2,
        "door": 2,
        "zipCode": "08010",
        "city": "Barcelona",
        "country": "España"
    }
})

db.glasses.insert({
    "_id": 1,
    "providerId": 1,
    "brand": "Lewis",
    "graduationR": 1.5,
    "graduationL": 1.5,
    "type": "metalica",
    "color": "verde",
    "precio": 45
})

db.clients.insert({
    "_id": 1,
    "name": "Juan",
    "address": "Calle Provenza 1, piso 3 puerta 2",
    "phone": "932121212",
    "email": "test@example.com",
    "createdAt": "2021-03-05",
    "recommendedClientId": 1,
    "orders": [
        {
            "seller": "Jorge",
            "glassesId": 1,
        },
        {
            "seller": "Susane",
            "glassesId": 20,
        }
    ]
})

