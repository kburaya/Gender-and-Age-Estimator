Age and Gender Estimator
--------------------
Client-server service, that predicts user age and gender. Client sends text messages to RabbitMQ, server(s) recieve messages and predict age/gender and print result to console or save to file.


### File list

1. :page_facing_up: **client.py** - sends messages to RabbitMQ.
2. :page_facing_up: **server.py** - receives messages from RabbitMQ, load model from file, predict age/gender and print answer to console/save to output file.
3. :page_facing_up: **input-data-handler**