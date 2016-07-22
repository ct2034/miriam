# cloudsim

## Requirements
Tested with Anylogic 7.3.2

http://www.anylogic.com/downloads

## joda-time Workaround
If Starting of the msb client leads to this error:

	java.lang.NoSuchMethodError: org.joda.time.format.DateTimeFormatter.withZoneUTC()Lorg/joda/time/format/DateTimeFormatter;

do this:

	cd bin/anylogic/plugins/com.anylogic.third_party_libraries_7.3.4.201605201443/lib/database/querydsl/
	mv joda-time-1.6.jar joda-time-1.6.jar_ORIG
	cp ~/src/cloudsim/JavaTest/joda-time-2.8.2.jar joda-time-1.6.jar
