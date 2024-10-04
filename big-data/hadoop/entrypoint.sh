#!/bin/bash

# Setting nodes in /opt/hadoop/etc/hadoop/core-site.xml
# Setting nodes in /opt/hadoop/etc/hadoop/hdfs-site.xml

# Install hadoop mapreduce
yarn jar share/hadoop/mapreduce/hadoop-mapreduce-examples-3.3.6.jar pi 10 15
