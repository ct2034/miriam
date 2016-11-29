# smartleitstand

[![Build Status](https://travis-ci.com/ct2034/smartleitstand.svg?token=h8cyxU2xaFZFLypWpbG5&branch=master)](https://travis-ci.com/ct2034/smartleitstand)

## Simulation

### Cloudsim
What can a Simulation do in the cloud?
![General](doc/cloud_sim_general.jpg?raw=true)

### Architecture
<!-- ![Concept](doc/anylogic.jpg?raw=true) -->

The following image illustrates the general concept of simulation in Anylogic and AGV sim:
![Concept](doc/160818_architecture_01.png?raw=true)

This shows the interconnection between both simulations
![Concept](doc/160818_architecture_02.png?raw=true)

## Anylogic

### Requirements
Tested with Anylogic ...
* 7.3.2
* 7.3.4

http://www.anylogic.com/downloads

### joda-time Workaround
If starting of the msb client leads to this error:

	java.lang.NoSuchMethodError: org.joda.time.format.DateTimeFormatter.withZoneUTC()Lorg/joda/time/format/DateTimeFormatter;

get the joda-time version, used in vfk client and manally download it: http://central.maven.org/maven2/joda-time/joda-time/2.8.2/ 

do this:

	cd bin/anylogic/plugins/com.anylogic.third_party_libraries_7.3.4.201605201443/lib/database/querydsl/
	mv joda-time-1.6.jar joda-time-1.6.jar_ORIG
	cp ~/src/cloudsim/JavaTest/joda-time-2.8.2.jar joda-time-1.6.jar



### Logger-Config-file / Debuggen with Eclipse
http://www.cs.usask.ca/faculty/ndo885/Classes/MIT15879/LectureSlides/Lecture%2023%20--%20Debugging%20in%20AnyLogic.pdf
