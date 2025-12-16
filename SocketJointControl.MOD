MODULE SocketJointControl
    CONST robtarget p_home:=[[202.83,79.22,-392.47],[0.579477,-0.0549491,-0.028588,0.812631],[0,0,1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
	LOCAL CONST robtarget p50:=[[10.07,18.66,10.95],[0.0280679,0.912412,0.407378,0.0275859],[0,-1,1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    VAR socketdev server_socket;
    VAR socketdev client_socket;
    VAR string cmd;
    VAR string cmd1;
    VAR num i;
    VAR num len;
    TASK PERS tooldata tool_pick_place:=[TRUE,[[12.4982,-1.4986,104.54],[1,0,0,0]],[3,[1,1,1],[1,0,0,0],0,0,0]];
    TASK PERS wobjdata wobj1:=[FALSE,TRUE,"",[[0,0,0],[1,0,0,0]],[[410.683,563.228,121.556],[0.0620808,-0.978161,0.0141766,0.197855]]];
    CONST robtarget pcamara:=[[-13.03,382.55,-275.60],[0.0346638,0.25427,0.0303543,0.966035],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pick13:=[[104.59,390.33,-0.85],[0.0316418,0.227059,0.0402709,0.972534],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pick13ac:=[[26.32,367.01,-184.85],[0.0316453,0.227061,0.0402688,0.972533],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pick12:=[[100.80,307.87,7.91],[0.0316168,0.227074,0.0402675,0.972531],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pick12ac:=[[41.05,293.49,-132.99],[0.0316259,0.227078,0.0402594,0.97253],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pick11:=[[99.88,228.34,14.62],[0.0315856,0.227135,0.0402316,0.972519],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pick11ac:=[[34.46,208.85,-139.20],[0.0315787,0.227135,0.0402296,0.97252],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pick21:=[[27.36,230.43,51.00],[0.0316892,0.227008,0.0402895,0.972543],[0,-1,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pick21ac:=[[-43.01,209.46,-114.44],[0.0316891,0.227015,0.0402848,0.972542],[0,-1,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pick22:=[[28.44,311.35,46.15],[0.0307709,0.218445,0.043151,0.974409],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pick22ac:=[[-34.99,292.31,-102.20],[0.0316914,0.226991,0.0402932,0.972547],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pick23:=[[33.33,390.82,32.61],[0.0317176,0.226958,0.0403076,0.972553],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pick23acc:=[[-55.31,364.41,-175.75],[0.0317147,0.226961,0.040301,0.972553],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pick24:=[[37.59,474.03,22.32],[0.0317097,0.226941,0.0403021,0.972558],[0,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pick24ac:=[[-53.81,446.81,-192.53],[0.031703,0.226941,0.0403013,0.972558],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pick14:=[[107.34,470.53,-12.49],[0.031681,0.227033,0.0402846,0.972538],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pick14ac:=[[29.95,447.47,-194.41],[0.0316876,0.227036,0.0402821,0.972537],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget placeac:=[[90.74,320.68,-255.46],[0.0307524,0.218472,0.0431349,0.974404],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget place:=[[135.22,333.93,-150.91],[0.0307591,0.218471,0.0431438,0.974404],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pickhome:=[[-136.96,422.04,-388.01],[0.0316977,0.226947,0.0402947,0.972557],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];    ! primer caracter del comando

    !========================
    ! MAIN: recibe secuencia
    !========================
    PROC main()
        MoveJ pcamara, v30, z1,  tool_pick_place\WObj:=wobj1;
        TPWrite "Creando socket...";
        SocketCreate server_socket;

        TPWrite "Haciendo bind al puerto 5000...";
        SocketBind server_socket, "", 5000;

        TPWrite "Escuchando conexiones entrantes...";
        SocketListen server_socket;

        TPWrite "Esperando conexion del PC...";
        SocketAccept server_socket, client_socket;

        TPWrite "Conexion establecida. Esperando SECUENCIA...";
        SocketReceive client_socket \Str:=cmd;

        TPWrite "Secuencia recibida: " + cmd;

        len := StrLen(cmd);

        FOR i FROM 1 TO len DO
            cmd1 := StrPart(cmd, i, 1);
            TPWrite "Paso " + NumToStr(i,0) + " -> " + cmd1;

            IF cmd1 = "1" THEN
                TPWrite "Ejecutando Pick_11";
                Pick_11;
                WaitTime 2;
                Place_ok;
                WaitTime 2;

            ELSEIF cmd1 = "2" THEN
                TPWrite "Ejecutando Pick_12";
                Pick_12;
                WaitTime 2;
                Place_ok;
                WaitTime 2;

            ELSEIF cmd1 = "3" THEN
                TPWrite "Ejecutando Pick_13";
                Pick_13;
                WaitTime 2;
                Place_ok;
                WaitTime 2;

            ELSEIF cmd1 = "4" THEN
                TPWrite "Ejecutando Pick_14";
                Pick_14;
                WaitTime 2;
                Place_ok;
                WaitTime 2;

            ELSEIF cmd1 = "5" THEN
                TPWrite "Ejecutando Pick_21";
                Pick_21;
                WaitTime 2;
                Place_ok;
                WaitTime 2;

            ELSEIF cmd1 = "6" THEN
                TPWrite "Ejecutando Pick_22";
                Pick_22;
                WaitTime 2;
                Place_ok;
                WaitTime 2;

            ELSEIF cmd1 = "7" THEN
                TPWrite "Ejecutando Pick_23";
                Pick_23;
                WaitTime 2;
                Place_ok;
                WaitTime 2;

            ELSEIF cmd1 = "8" THEN
                TPWrite "Ejecutando Pick_24";
                Pick_24;
                WaitTime 2;
                Place_ok;
                WaitTime 2;

            ELSEIF cmd1 = "9" THEN
                TPWrite "Fin de secuencia.";
                EXIT;

            ELSE
                TPWrite "Caracter desconocido: " + cmd1;
            ENDIF
        ENDFOR

        SocketClose client_socket;
        SocketClose server_socket;
    ENDPROC



    PROC Pick_11()
        MoveJ pick11ac  , v30, z1, tool_pick_place\WObj:=wobj1;
        MoveL pick11  , v30, z1, tool_pick_place\WObj:=wobj1;
        WaitTime 2;
        Set DO_04;
        WaitTime 2;
    ENDPROC
    PROC Pick_12()
        MoveJ pick12ac  , v30, z1, tool_pick_place\WObj:=wobj1;
        MoveL pick12  , v30, z1, tool_pick_place\WObj:=wobj1;
        WaitTime 2;
        Set DO_04;
        WaitTime 2;
    ENDPROC
    PROC Pick_13()
        MoveJ pick13ac  , v30, z1, tool_pick_place\WObj:=wobj1;
        MoveL pick13  , v30, z1, tool_pick_place\WObj:=wobj1;
        WaitTime 2;
        Set DO_04;
        WaitTime 2;
    ENDPROC
    PROC Pick_14()
        MoveJ pick14ac  , v30, z1, tool_pick_place\WObj:=wobj1;
        MoveL pick14  , v30, z1, tool_pick_place\WObj:=wobj1;
        WaitTime 2;
        Set DO_04;
        WaitTime 2;
    ENDPROC
    
    PROC Pick_21()
        MoveJ pick21ac  , v30, z1, tool_pick_place\WObj:=wobj1;
        MoveL pick21  , v30, z1, tool_pick_place\WObj:=wobj1;
        WaitTime 2;
        Set DO_04;
        WaitTime 2;
    ENDPROC
    PROC Pick_22()
        MoveJ pick22ac  , v30, z1, tool_pick_place\WObj:=wobj1;
        MoveL pick22  , v30, z1, tool_pick_place\WObj:=wobj1;
        WaitTime 2;
        Set DO_04;
        WaitTime 2;
    ENDPROC
    PROC Pick_23()
        MoveJ pick23acc  , v30, z1, tool_pick_place\WObj:=wobj1;
        MoveL pick23  , v30, z1, tool_pick_place\WObj:=wobj1;
        WaitTime 2;
        Set DO_04;
        WaitTime 2;
    ENDPROC
    PROC Pick_24()
        MoveJ pick24ac  , v30, z1, tool_pick_place\WObj:=wobj1;
        MoveL pick24  , v30, z1, tool_pick_place\WObj:=wobj1;
        WaitTime 2;
        Set DO_04;
        WaitTime 2;
    ENDPROC

    PROC Place_ok()
        MoveJ pcamara, v30, z1,  tool_pick_place\WObj:=wobj1;
        MoveJ placeac, v30, z1,  tool_pick_place\WObj:=wobj1;
        MoveL place, v30, z1,  tool_pick_place\WObj:=wobj1;
        WaitTime 2;
        ReSet DO_04;
        WaitTime 2;
        MoveJ pcamara, v30, z1,  tool_pick_place\WObj:=wobj1;
    ENDPROC

ENDMODULE