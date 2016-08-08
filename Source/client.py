import pika
import sys

#declare rabbitMQ queue for messages sending
connection = pika.BlockingConnection(pika.ConnectionParameters(
        host='localhost'))
channel = connection.channel()
channel.queue_declare(queue='messages_queue', durable=True)
