const DB_NAME = 'mongotest';

// The current database to use.
use(DB_NAME);

/*1.1. Escriu una consulta per mostrar tots els documents en la col.lecció Restaurants*/
/*Resultado = 3772 documentos*/
db.restaurants.find();

/*1.2. Escriu una consulta per mostrar el restaurant_id, name, borough i cuisine per tots els 
documents en la col.lecció Restaurants*/
/*El _id y el restaurant_id no son lo mismo. El _id no está especificado pero lo incluye igual.
/*Resultado = 3772 documentos*/
db.restaurants.find({}, { restaurant_id: 1, name: 1, borough: 1, cuisine: 1 })

/*1.3. Escriu una consulta per mostrar el restaurant_id, name, borough i cuisine, però excloure 
el camp _id per tots els documents en la col.lecció Restaurants*/
/*Resultado = 3772 documentos*/
db.restaurants.find({}, { _id: 0, restaurant_id: 1, name: 1, borough: 1, cuisine: 1 })

/*1.4. Escriu una consulta per mostrar restaurant_id, name, borough i zip code, però excloure el camp
 _id per tots els documents en la col.lecció Restaurants*/
/*El campo zipcode está anidado dentro del campo address y cambia la sintaxis.*/
/*Resultado 3772 documentos)*/
db.restaurants.find({}, { _id: 0, restaurant_id: 1, name: 1, borough: 1, address: { zipcode: 1 } })

/*1.5. Escriu una consulta per mostrar tot els restaurants que estan en el Bronx*/
/*Resultado 309 documentos)*/
db.restaurants.find({ borough: "Bronx" })

/*1.6. Escriu una consulta per mostrar els primers 5 restaurants que estan en el Bronx*/
/*Resultado los primeros 5 documentos)*/
db.restaurants.find({ borough: "Bronx" }).limit(5)

/*1.7. Escriu una consulta per mostrar el pròxim 5 restaurants després de saltar els primers 5 del Bronx*/
/*Resultado los siguientes 5 documentos)*/
db.restaurants.find({ borough: "Bronx" }).skip(5).limit(5)

/*1.8. Escriu una consulta per trobar els restaurants que tenen un score de m s de 90*/
/*Resultado 3 documentos)*/
db.restaurants.find({ 'grades.score': { $gt: 90 } })

/*1.9. Escriu una consulta per trobar els restaurants que tenen un score de més que 80 però menys que 100*/
/*Resultado 4 documentos)*/
db.restaurants.find({ 'grades.score': { $gt: 80, $lt: 100 } })

/*1.10. Escriu una consulta per trobar els restaurants quins localitzen en valor de latitud menys que -95.754168*/
/*Resultado 3 documentos)*/
db.restaurants.find({ 'address.coord.0': { $lt: -95.754168 } })

/*1.11. Escriu una consulta de MongoDB per a trobar els restaurants que no preparen cap cuisine de 'American' i 
el seu puntaje de qualificació superior a 70 i latitud inferior a -65.754168*/
/*La palabra "American" tiene un espacio detrás y la búsqueda hay que hacerla por "American "
/*Resultado 5 documentos)*/
db.restaurants.find({ $and: [
	{cuisine: { $ne: "American " }},
	{'grades.score': { $gt: 70 }},
	{'address.coord.1': { $lt: -65.754168 }}
]})

/*1.12. Escriu una consulta per trobar els restaurants quins no preparen cap cuisine de 'American' i va 
aconseguir un marcador més que 70 i localitzat en la longitud menys que -65.754168. 
Nota : Fes aquesta consulta sense utilitzar $and operador*/
/*Resultado 5 documentos)*/
db.restaurants.find({ $and : [{"cuisine": { $ne : "American "} }, { "address.coord.1": { $lt: -65.754168} }, { "grades.score" : { $gt : 70 } }]})

/*2.1. Escriu una consulta per trobar els restaurants quins no preparen cap cuisine de 'American ' i 
va aconseguir un punt de grau 'A' no pertany a Brooklyn. S'ha de mostrar el document segons la cuisine en ordre descendent*/
/*Resultado 2017 documentos*/
db.restaurants.find({ cuisine: { $ne: "American " }, 'grades.grade': 'A', borough: { $ne: 'Brooklyn' }}).sort({ cuisine: -1 })

/*2.2. Escriu una consulta per trobar el restaurant_id, name, borough i cuisine per a aquells restaurants 
quin contenir 'Wil' com les tres primeres lletres en el seu nom*/
/*Resultado 3 documentos*/
db.restaurants.find( { name: {"$regex": "^Wil"}}, { restaurant_id:1, name:1, borough:1, cuisine: 1, _id:0 } )

/*2.3. Escriu una consulta per trobar el restaurant_id, name, borough i cuisine per a aquells 
restaurants quin contenir 'ces' com les últimes tres lletres en el seu nom*/
/*Resultado 6 documentos*/
db.restaurants.find( { name: {"$regex": "ces$"}}, {restaurant_id:1, name:1, borough:1, cuisine:1});

/*2.4. Escriu una consulta per trobar el restaurant_id, name, borough i cuisine per a aquells restaurants quin 
contenir 'Reg' com tres lletres en algun lloc en el seu nom*/
/*Resultado 7 documentos*/
db.restaurants.find( { name: {"$regex": "Reg"}}, {restaurant_id:1, name:1, borough:1, cuisine:1})

/*2.5. Escriu una consulta per trobar els restaurants quins pertanyen al Bronx i va preparar qualsevol plat 
American o xinés. "American " va con un espacio al final*/
/*Resultado 91 documentos*/
db.restaurants.find({ cuisine: { $in: ["American ", "Chinese"] }, borough: "Bronx" })

/*2.6. Escriu una consulta per trobar el restaurant_id, name, borough i cuisine per a aquells restaurants 
que pertanyen a Staten Island o Queens o Bronx or Brooklyn*/
/*Resultado 1889 docuemntos*/
db.restaurants.find({ borough: {"$in": ["Staten Island", "Queens", "Bronx", "Brooklyn"]}}, {restaurant_id:1, name:1, borough:1, cuisine:1})

/*2.7. Escriu una consulta per trobar el restaurant_id, name, borough i cuisine per a aquells restaurants 
que no pertanyen a Staten Island o Queens o Bronx or Brooklyn*/
/*Resultado 1883 documentos*/
db.restaurants.find({ borough: {"$nin": ["Staten Island", "Queens", "Bronx", "Brooklyn"]}}, {restaurant_id:1, name:1, borough:1, cuisine:1})

/*2.8. Escriu una consulta per trobar el restaurant_id, name, borough i cuisine per a aquells restaurants 
que aconsegueixin un marcador quin no és més que 10*/
/*Resultado 3529 documentos */ 
db.restaurants.find({ 'grades.score': { $lte: 10 } }, {restaurant_id:1, name:1, borough:1, cuisine:1})

/*2.9. Escriu una consulta per trobar el restaurant_id, name, borough i cuisine per a aquells restaurants que 
preparen peix excepte 'American ' i 'Chinese' o el name del restaurant comença amb lletres 'Wil'*/
/*Resultado 2402 documentos */ 
db.restaurants.find({ $or: [
	{cuisine: { $nin: ["American ", "Chinese"] } },
	{ name: { $regex: "^Wil" } }
]}, {restaurant_id:1, name:1, borough:1, cuisine:1})

/*2.10. Escriu una consulta per trobar el restaurant_id, name, i grades per a aquells restaurants que aconsegueixin 
un grau "A" i un score 11 en dades d'estudi ISODate "2014-08-11T00:00:00Z"*/
/*Resultado 0 documentos */ 
db.restaurants.find( { grades: { $elemMatch: { grade:'A', score: 11, date: ISODate ("2014-08-11T00:00:00Z") } } }, 
			{ restaurant_id: 1, name: 1, grades: 1, _id: 0 } )
db.restaurants.find({grades: {$all: [{$elemMatch: {"grade": "A", "score": 11, "date": ISODate("2014-08-11T00:00:00Z")}}]}}, 
			{_id: 0, restaurant_id: "$restaurant_id", name: "$name", grades: "$grades"})
db.restaurants.find({
		$and:[
			{"grades.date": ISODate("2014-08-11T00:00:00Z")},
			{"grades.grade": "A"},
			{"grades.score": 10}
		]},
		{restaurant_id:1, name:1, grades:1}
	);

db.restaurants.find( {  grades :  { date :  ISODate("2014-08-11T00:00:00Z") , grade : "A", score : 11  } } ,
           {_id :0, restaurant_id:1, name: 1, borough: 1, cuisine: 1 , grades:1})


/*3.1. Escriu una consulta per trobar el restaurant_id, name i grades per a aquells restaurants on el 2n element 
de varietat de graus conté un grau de "A" i marcador 9 sobre un ISODate "2014-08-11T00:00:00Z"*/
/*Resultado 0 documentos*/
db.restaurants.find( { 'grades.1': { $elemMatch: { grade:'A', score: 9, date: ISODate ("2014-08-11T00:00:00Z") } } }, 
		         	{ restaurant_id: 1, name: 1, grades: 1, _id: 0 } )

db.restaurants.find({$and: [{"grades.1.grade": "A"}, {"grades.1.score": 9}, 
	   {"grades.1.date": ISODate("2014-08-11T00:00:00Z")}]}, 
       {_id: 0, restaurant_id: "$restaurant_id", name: "$name", grades: "$grades"})
 
db.restaurants.find({
		$and:[
			{"grades.1.grade": "A"},
			{"grades.score": 9},
			{"grades.date": ISODate("2014-08-11T00:00:00Z")}
		]},
		{restaurant_id:1, name:1, grades:1}
	);

db.restaurants.find( {  "grades.1" :  { date :  ISODate("2014-08-11T00:00:00Z") , grade : "A", score : 9  } } ,
           {_id :0, restaurant_id:1, name: 1,  grades:1 }  );

/*3.2. Escriu una consulta per trobar el restaurant_id, name, adress i ubicacio geografica per a aquells restaurants 
on el segon element del array coord cont  un valor quin es mes que 42 i fins a 52*/
/*Resultado 7 documentos*/
db.restaurants.find(
	{ "address.coord.1": { $gt: 42, $lt: 52 } },
	{_id :0, restaurant_id: 1, name: 1, address: 1 }
)

/*3.3. Escriu una consulta per organitzar el nom dels restaurants en ordre ascendent juntament amb totes les columnes*/
/*Resultado 3772 documentos*/
db.restaurants.find().sort({ name: 1 })

/*3.4. Escriu una consulta per organitzar el nom dels restaurants en descendir juntament amb totes les columnes*/
/*Resultado 3772 docuemntos. El primero (DESC) es Zum Stammtisch*/
db.restaurants.find().sort({ name: -1 })

/*3.5. Escriu una consulta a organitzar el nom de la cuisine en ordre ascendent i per el mateix barri de cuisine. 
Ordre descendint*/
/*Resultado 3772 documentos*/ 
db.restaurants.find().sort({ cuisine: 1, 'address.street': -1 })

/*3.6. Escriu una consulta per saber tant si totes les direccions contenen el carrer o no*/
/*Resultado 0 documentos*/ 
db.restaurants.find({ 'address.street': { $exists: false } })

/*3.7. Escriu una consulta quin seleccionarà tots el documents en la col.lecció de restaurants on el valor del camp coord és 
Double*/
/*Resultado 3772 documentos*/
db.restaurants.find({ 'address.coord': { $type : 1 } })

/*3.8. Escriu una consulta quin seleccionarà el restaurant_id, name i grade per a aquells restaurants quins retorns 0 com a 
resta després de dividir el marcador per 7*/
/*Resultado 1585 documentos*/
db.restaurants.find({ "grades.score" : { $mod : [7, 0] } }, { restaurant_id: 1, name: 1, grades: 1 })

/*3.9. Escriu una consulta per trobar el name de restaurant, borough, longitud i altitud i cuisine per a aquells restaurants 
que contenen 'mon' com tres lletres en algun lloc del seu name*/
/*Resultado 32 o 21 documentos si no se consideran mayusculas*/
db.restaurants.find({ name: { $regex: /mon/ } }, { name: 1, borough: 1, "address.coord": 1, cuisine:1 })

/*3.10. Escriu una consulta per trobar el name de restaurant, borough, longitud i latitud i cuisine per a aquells restaurants 
que conteinen 'Mad' com primeres tres lletres del seu name*/
/*Resultado 8 documentos*/
db.restaurants.find({ name: { $regex: /^Mad.*/ } },{ name: 1, borough: 1, "address.coord": 1, cuisine: 1 })
