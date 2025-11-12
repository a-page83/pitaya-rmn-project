CFLAGS  = -std=gnu11 -Wall ## -Werror
CFLAGS += -I/opt/redpitaya/include
LDFLAGS = -L/opt/redpitaya/lib
LDLIBS = -static -lrp-hw-can -lrp -lrp-hw-calib -lrp-hw-profiles

INCLUDE += -I/opt/redpitaya/include/api250-12
LDLIBS += -lrp-gpio -lrp-i2c
LDLIBS += -lrp-hw -lm -lstdc++ -lpthread -li2c -lsocketcan

# Si l'argument DEBUG est passé
ifeq ($(DEBUG), 1)
    CFLAGS += -g
endif

# List of compiled object files (not yet linked to executable)

PRGS =  sin\
	Burst\
	Acquisition_axi\
	digital_led_blink\
	Simulation\
	SimulationV1\
	Find_excitation_time\
	Find_Frequency\

OBJS := $(patsubst %,%.o,$(PRGS))
SRC := $(patsubst %,%.c,$(PRGS))


all: $(PRGS)



$(PRGS): %: %.c functions.o
	$(CC) $< $(CFLAGS) $(LDFLAGS) $(LDLIBS) -o $@.exe functions.o
	
# Compile functions.c into an object so programs can link against it
functions.o: functions.c functions.h
	$(CC) -c functions.c $(CFLAGS) -o functions.o

clean:
	$(RM) *.o
	$(RM) $(OBJS)

clean_all: clean
	$(RM) $(PRGS)
	$(RM) *.exe

debug:
	$(MAKE) DEBUG=1