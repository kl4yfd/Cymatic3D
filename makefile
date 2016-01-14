current: 
	@echo "[cymatic3d build]: please use one of the following configurations:"; echo "   make linux-alsa, make linux-jack, make linux-oss, make osx, or make win32"

install:
	@ cd src/cymatic3d/ ; cp cymatic3d /usr/local/bin/; chmod 755 /usr/local/bin/cymatic3d

osx: 
	@ cd src/cymatic3d/ ; make -f makefile.osx

osx-ub:
	@ cd src/cymatic3d/ ;  make -f makefile.osx-ub

linux-oss: 
	@ cd src/cymatic3d/ ;  make -f makefile.oss 

linux-jack:
	@ cd src/cymatic3d/ ;  make -f makefile.jack

linux-alsa: 
	@ cd src/cymatic3d/ ;  make -f makefile.alsa

win32: 
	@ cd src/cymatic3d/ ;  make -f makefile.win32

clean:
	@ cd src/cymatic3d/ ;  rm -f *.o cymatic3d cymatic3d.exe
