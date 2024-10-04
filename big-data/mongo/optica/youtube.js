const database = 'pizzeria';

use(database);

db.createCollection('users');
db.createCollection('videos');
db.createCollection('channels');
db.createCollection('tags');
db.createCollection('subscriptions');
db.createCollection('video_likes');
db.createCollection('playlists');
db.createCollection('comments');
db.createCollection('comment_likes');

db.users.insert({
    "_id": 1,
    "email": "user@example.com",
    "password": "hashed_password",
    "username": "user123",
    "birthdate": "1990-01-01T00:00:00Z",
    "gender": "male",
    "country": "Spain",
    "postal_code": "08001"
})

db.videos.insert({
    "_id": 1,
    "title": "How to Code in Python",
    "description": "This video teaches you how to code in Python.",
    "size": 104857600,
    "file_name": "python_tutorial.mp4",
    "duration": 600,
    "thumbnail": "thumbnail.jpg",
    "views": 1024,
    "likes": 100,
    "dislikes": 5,
    "status": "public",
    "user_id": 1,
    "upload_datetime": "2023-10-04T12:30:00Z",
    "tags": [
        1,
        2
    ]
})

db.channels.insert({
    "_id": 1,
    "name": "TechChannel",
    "description": "Channel about tech tutorials",
    "creation_date": "2023-01-01T00:00:00Z",
    "user_id": 1
})

db.tags.insert({
    "_id": 1,
    "name": "Python"
})

db.subscriptions.insert({
    "_id": 1,
    "subscriber_id": 1,
    "channel_id": 1,
    "subscription_date": "2023-10-04T16:00:00Z"
})

db.video_likes.insert({
    "video_id": 1,
    "user_id": 1,
    "type": "like",
    "datetime": "2023-10-04T15:00:00Z"
})

db.playlists.insert({
    "_id": 1,
    "name": "Favorite Coding Tutorials",
    "creation_date": "2023-10-01T00:00:00Z",
    "status": "public",
    "user_id": 1,
    "videos": [
        1,
        2
    ]
})

db.comments.insert({
    "_id": 1,
    "video_id": 1,
    "user_id": 1,
    "text": "Great tutorial!",
    "comment_datetime": "2023-10-04T14:30:00Z"
})

db.comment_likes.insert({
    "comment_id": 1,
    "user_id": 1,
    "type": "like",
    "datetime": "2023-10-04T15:00:00Z"
})

