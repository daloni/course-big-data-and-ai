const database = 'pizzeria';

use(database);

db.createCollection('users');

db.users.insert({
    "_id": 1,
    "email": "user@example.com",
    "password": "hashed_password",
    "username": "user123",
    "birthdate": "1990-01-01T00:00:00Z",
    "gender": "male",
    "country": "Spain",
    "postal_code": "08001",
    "account_type": "premium"
})

db.subscriptions.insert({
    "_id": 1,
    "user_id": 1,
    "start_date": "2023-01-01T00:00:00Z",
    "renewal_date": "2024-01-01T00:00:00Z",
    "payment_method": {
        "type": "credit_card",
        "details": {
            "card_number": "1234-5678-9876-5432",
            "expiry_month": 12,
            "expiry_year": 2025,
            "security_code": "123"
        }
    }
})

db.payments.insert({
    "_id": 1,
    "user_id": 1,
    "payment_date": "2023-01-01T00:00:00Z",
    "order_number": "ORD-12345678",
    "total_amount": 9.99
})

db.playlists.insert({
    "_id": 1,
    "user_id": 1,
    "title": "My Favorite Songs",
    "song_count": 25,
    "creation_date": "2023-05-01T00:00:00Z",
    "deleted": false,
    "deleted_date": null,
    "status": "active",
    "shared": true,
    "songs": [
        {
            "song_id": 1,
            "added_by": 1,
            "added_date": "2023-05-01T00:00:00Z"
        }
    ]
})

db.songs.insert({
    "_id": 1,
    "title": "Song Title",
    "duration": 180,
    "play_count": 1024,
    "album_id": 1,
    "artist_id": 1
})

db.albums.insert({
    "_id": 1,
    "title": "Album Title",
    "release_year": 2020,
    "cover_image": "album_cover.jpg",
    "artist_id": 1,
    "songs": [
        1,
        2,
    ]
})

db.artists.insert({
    "_id": 1,
    "name": "Artist Name",
    "image": "artist_image.jpg"
})

db.follows.insert({
    "_id": 1,
    "user_id": 1,
    "artist_id": 1,
    "follow_date": "2023-06-01T00:00:00Z"
})

db.related_artists.insert({
    "_id": 1,
    "artist_id": 1,
    "related_artist_id": 1
})

db.favorites.insert({
    "_id": 1,
    "user_id": 1,
    "type": "song",
    "item_id": 1,
    "favorite_date": "2023-07-01T00:00:00Z"
})
