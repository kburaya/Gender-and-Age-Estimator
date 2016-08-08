#!/usr/bin/env python
import pika
import time

def callback(ch, method, properties, body):
    print (" [x] Received %r" % (body,))
    #TODO message processing here!
    print (" [x] Done")
    ch.basic_ack(delivery_tag = method.delivery_tag)

def main():
    connection = pika.BlockingConnection(pika.ConnectionParameters(
            host='localhost'))
    channel = connection.channel()

    channel.queue_declare(queue='messages_queue', durable=True)
    print (' [*] Waiting for messages. To exit press CTRL+C')

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(callback,
                          queue='messages_queue')

    channel.start_consuming()