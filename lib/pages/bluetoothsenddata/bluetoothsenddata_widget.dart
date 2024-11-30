import '/backend/schema/structs/index.dart';
import '/flutter_flow/flutter_flow_icon_button.dart';
import '/flutter_flow/flutter_flow_theme.dart';
import '/flutter_flow/flutter_flow_util.dart';
import '/flutter_flow/flutter_flow_widgets.dart';
import '/flutter_flow/instant_timer.dart';
import '/widgets/strength_indicator/strength_indicator_widget.dart';
import '/custom_code/actions/index.dart' as actions;
import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:provider/provider.dart';
import 'bluetoothsenddata_model.dart';
export 'bluetoothsenddata_model.dart';

class BluetoothsenddataWidget extends StatefulWidget {
  const BluetoothsenddataWidget({
    super.key,
    required this.deviceName,
    required this.deviceId,
    required this.deviceRssi,
    required this.hasWriteCharacteristic,
  });

  final String? deviceName;
  final String? deviceId;
  final int? deviceRssi;
  final bool? hasWriteCharacteristic;

  @override
  State<BluetoothsenddataWidget> createState() =>
      _BluetoothsenddataWidgetState();
}

class _BluetoothsenddataWidgetState extends State<BluetoothsenddataWidget> {
  late BluetoothsenddataModel _model;

  final scaffoldKey = GlobalKey<ScaffoldState>();

  @override
  void initState() {
    super.initState();
    _model = createModel(context, () => BluetoothsenddataModel());

    // On page load action.
    SchedulerBinding.instance.addPostFrameCallback((_) async {
      _model.currentRssi = widget!.deviceRssi;
      safeSetState(() {});
      _model.rssiUpdateTimer = InstantTimer.periodic(
        duration: Duration(milliseconds: 2000),
        callback: (timer) async {
          _model.updatedRssi = await actions.getRssi(
            BTDeviceStruct(
              name: widget!.deviceName,
              id: widget!.deviceId,
              rssi: widget!.deviceRssi,
            ),
          );
          _model.currentRssi = _model.updatedRssi;
          safeSetState(() {});
        },
        startImmediately: true,
      );
    });

    WidgetsBinding.instance.addPostFrameCallback((_) => safeSetState(() {}));
  }

  @override
  void dispose() {
    _model.dispose();

    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () => FocusScope.of(context).unfocus(),
      child: Scaffold(
        key: scaffoldKey,
        backgroundColor: FlutterFlowTheme.of(context).primaryBackground,
        appBar: AppBar(
          backgroundColor: FlutterFlowTheme.of(context).primary,
          iconTheme: IconThemeData(
              color: FlutterFlowTheme.of(context).secondaryBackground),
          automaticallyImplyLeading: true,
          title: Column(
            mainAxisSize: MainAxisSize.max,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisSize: MainAxisSize.max,
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Padding(
                    padding:
                        EdgeInsetsDirectional.fromSTEB(0.0, 0.0, 10.0, 0.0),
                    child: Text(
                      widget!.deviceName!,
                      style: FlutterFlowTheme.of(context).bodyLarge.override(
                            fontFamily: 'Montserrat',
                            color: FlutterFlowTheme.of(context).alternate,
                            fontSize: 18.0,
                            letterSpacing: 0.0,
                          ),
                    ),
                  ),
                  if (_model.currentRssi != null)
                    wrapWithModel(
                      model: _model.strengthIndicatorModel,
                      updateCallback: () => safeSetState(() {}),
                      child: StrengthIndicatorWidget(
                        rssi: _model.currentRssi!,
                        color: valueOrDefault<Color>(
                          () {
                            if (_model.currentRssi! >= -67) {
                              return FlutterFlowTheme.of(context).success;
                            } else if (_model.currentRssi! >= -90) {
                              return FlutterFlowTheme.of(context).warning;
                            } else {
                              return FlutterFlowTheme.of(context).error;
                            }
                          }(),
                          FlutterFlowTheme.of(context).success,
                        ),
                      ),
                    ),
                ],
              ),
              Padding(
                padding: EdgeInsetsDirectional.fromSTEB(0.0, 5.0, 0.0, 0.0),
                child: Text(
                  widget!.deviceId!,
                  style: FlutterFlowTheme.of(context).bodyMedium.override(
                        fontFamily: 'Montserrat',
                        color: FlutterFlowTheme.of(context).alternate,
                        fontSize: 12.0,
                        letterSpacing: 0.0,
                      ),
                ),
              ),
            ],
          ),
          actions: [
            Container(
              height: 200.0,
              decoration: BoxDecoration(),
            ),
            Padding(
              padding: EdgeInsetsDirectional.fromSTEB(0.0, 0.0, 10.0, 0.0),
              child: Row(
                mainAxisSize: MainAxisSize.max,
                children: [
                  FlutterFlowIconButton(
                    borderColor: Colors.transparent,
                    borderRadius: 20.0,
                    buttonSize: 45.0,
                    icon: Icon(
                      Icons.bluetooth_disabled_rounded,
                      color: FlutterFlowTheme.of(context).error,
                      size: 30.0,
                    ),
                    onPressed: () async {
                      await actions.disconnectDevice(
                        BTDeviceStruct(
                          name: widget!.deviceName,
                          id: widget!.deviceId,
                          rssi: _model.currentRssi,
                        ),
                      );
                      context.safePop();
                    },
                  ),
                ],
              ),
            ),
          ],
          centerTitle: false,
          elevation: 0.0,
        ),
        body: SafeArea(
          top: true,
          child: Padding(
            padding: EdgeInsetsDirectional.fromSTEB(16.0, 16.0, 16.0, 0.0),
            child: Column(
              mainAxisSize: MainAxisSize.max,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Align(
                  alignment: AlignmentDirectional(1.0, -1.0),
                  child: InkWell(
                    splashColor: Colors.transparent,
                    focusColor: Colors.transparent,
                    hoverColor: Colors.transparent,
                    highlightColor: Colors.transparent,
                    onTap: () async {
                      context.pushNamed(
                        'login',
                        queryParameters: {
                          'deviceName': serializeParam(
                            '',
                            ParamType.String,
                          ),
                          'deviceId': serializeParam(
                            '',
                            ParamType.String,
                          ),
                          'deviceRssi': serializeParam(
                            0,
                            ParamType.int,
                          ),
                          'hasWriteCharacteristic': serializeParam(
                            false,
                            ParamType.bool,
                          ),
                        }.withoutNulls,
                      );
                    },
                    child: RichText(
                      textScaler: MediaQuery.of(context).textScaler,
                      text: TextSpan(
                        children: [
                          TextSpan(
                            text: 'Log out',
                            style: FlutterFlowTheme.of(context)
                                .bodyMedium
                                .override(
                                  fontFamily: 'Montserrat',
                                  color: FlutterFlowTheme.of(context).primary,
                                  letterSpacing: 0.0,
                                  fontWeight: FontWeight.bold,
                                ),
                          )
                        ],
                        style: FlutterFlowTheme.of(context).bodyMedium.override(
                              fontFamily: 'Plus Jakarta Sans',
                              letterSpacing: 0.0,
                            ),
                      ),
                      textAlign: TextAlign.center,
                    ),
                  ),
                ),
                Padding(
                  padding: EdgeInsetsDirectional.fromSTEB(0.0, 8.0, 0.0, 8.0),
                  child: Text(
                    'You can control the robot using the commands below.',
                    style: FlutterFlowTheme.of(context).labelMedium.override(
                          fontFamily: 'Plus Jakarta Sans',
                          letterSpacing: 0.0,
                          lineHeight: 1.4,
                        ),
                  ),
                ),
                Padding(
                  padding: EdgeInsetsDirectional.fromSTEB(0.0, 12.0, 0.0, 0.0),
                  child: Container(
                    width: 386.0,
                    height: 386.0,
                    decoration: BoxDecoration(
                      color: FlutterFlowTheme.of(context).primaryBackground,
                    ),
                    child: GridView(
                      padding: EdgeInsets.zero,
                      gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
                        crossAxisCount: 3,
                        crossAxisSpacing: 10.0,
                        mainAxisSpacing: 10.0,
                        childAspectRatio: 1.0,
                      ),
                      scrollDirection: Axis.vertical,
                      children: [
                        ClipRRect(
                          borderRadius: BorderRadius.circular(8.0),
                          child: Image.asset(
                            Theme.of(context).brightness == Brightness.dark
                                ? 'assets/images/Colorcard-hex_14181B-www.colorbook.io.png'
                                : 'assets/images/f1f4f8.png',
                            width: 300.0,
                            height: 300.0,
                            fit: BoxFit.cover,
                          ),
                        ),
                        FFButtonWidget(
                          onPressed: () async {
                            await actions.sendForward(
                              BTDeviceStruct(
                                name: widget!.deviceName,
                                id: widget!.deviceId,
                                rssi: widget!.deviceRssi,
                              ),
                              valueOrDefault<String>(
                                '',
                                '1',
                              ),
                            );
                          },
                          text: 'Forward',
                          options: FFButtonOptions(
                            width: 380.0,
                            height: 60.0,
                            padding: EdgeInsetsDirectional.fromSTEB(
                                16.0, 0.0, 16.0, 0.0),
                            iconPadding: EdgeInsetsDirectional.fromSTEB(
                                0.0, 0.0, 0.0, 0.0),
                            color: FlutterFlowTheme.of(context).primary,
                            textStyle: FlutterFlowTheme.of(context)
                                .titleSmall
                                .override(
                                  fontFamily: 'Montserrat',
                                  color: Colors.white,
                                  letterSpacing: 0.0,
                                ),
                            elevation: 0.0,
                            borderRadius: BorderRadius.circular(8.0),
                          ),
                        ),
                        ClipRRect(
                          borderRadius: BorderRadius.circular(8.0),
                          child: Image.asset(
                            Theme.of(context).brightness == Brightness.dark
                                ? 'assets/images/Colorcard-hex_14181B-www.colorbook.io.png'
                                : 'assets/images/f1f4f8.png',
                            width: 200.0,
                            height: 200.0,
                            fit: BoxFit.cover,
                          ),
                        ),
                        FFButtonWidget(
                          onPressed: () async {
                            await actions.sendLeft(
                              BTDeviceStruct(
                                name: widget!.deviceName,
                                id: widget!.deviceId,
                                rssi: widget!.deviceRssi,
                              ),
                              valueOrDefault<String>(
                                '',
                                '3',
                              ),
                            );
                          },
                          text: 'Left',
                          options: FFButtonOptions(
                            width: 380.0,
                            height: 60.0,
                            padding: EdgeInsetsDirectional.fromSTEB(
                                16.0, 0.0, 16.0, 0.0),
                            iconPadding: EdgeInsetsDirectional.fromSTEB(
                                0.0, 0.0, 0.0, 0.0),
                            color: FlutterFlowTheme.of(context).primary,
                            textStyle: FlutterFlowTheme.of(context)
                                .titleSmall
                                .override(
                                  fontFamily: 'Montserrat',
                                  color: Colors.white,
                                  letterSpacing: 0.0,
                                ),
                            elevation: 0.0,
                            borderRadius: BorderRadius.circular(8.0),
                          ),
                        ),
                        FFButtonWidget(
                          onPressed: () async {
                            context.pushNamed(
                              'recorder',
                              queryParameters: {
                                'deviceName': serializeParam(
                                  '',
                                  ParamType.String,
                                ),
                                'deviceId': serializeParam(
                                  '',
                                  ParamType.String,
                                ),
                                'deviceRssi': serializeParam(
                                  0,
                                  ParamType.int,
                                ),
                                'hasWriteCharacteristic': serializeParam(
                                  false,
                                  ParamType.bool,
                                ),
                              }.withoutNulls,
                            );
                          },
                          text: 'Voice',
                          options: FFButtonOptions(
                            height: 40.0,
                            padding: EdgeInsetsDirectional.fromSTEB(
                                16.0, 0.0, 16.0, 0.0),
                            iconPadding: EdgeInsetsDirectional.fromSTEB(
                                0.0, 0.0, 0.0, 0.0),
                            color: FlutterFlowTheme.of(context).primary,
                            textStyle: FlutterFlowTheme.of(context)
                                .titleSmall
                                .override(
                                  fontFamily: 'Montserrat',
                                  color: Colors.white,
                                  letterSpacing: 0.0,
                                ),
                            elevation: 0.0,
                            borderRadius: BorderRadius.circular(8.0),
                          ),
                        ),
                        FFButtonWidget(
                          onPressed: () async {
                            await actions.sendRight(
                              BTDeviceStruct(
                                name: widget!.deviceName,
                                id: widget!.deviceId,
                                rssi: widget!.deviceRssi,
                              ),
                              valueOrDefault<String>(
                                '',
                                '4',
                              ),
                            );
                          },
                          text: 'Right',
                          options: FFButtonOptions(
                            width: 380.0,
                            height: 60.0,
                            padding: EdgeInsetsDirectional.fromSTEB(
                                16.0, 0.0, 16.0, 0.0),
                            iconPadding: EdgeInsetsDirectional.fromSTEB(
                                0.0, 0.0, 0.0, 0.0),
                            color: FlutterFlowTheme.of(context).primary,
                            textStyle: FlutterFlowTheme.of(context)
                                .titleSmall
                                .override(
                                  fontFamily: 'Montserrat',
                                  color: Colors.white,
                                  letterSpacing: 0.0,
                                ),
                            elevation: 0.0,
                            borderRadius: BorderRadius.circular(8.0),
                          ),
                        ),
                        ClipRRect(
                          borderRadius: BorderRadius.circular(8.0),
                          child: Image.asset(
                            Theme.of(context).brightness == Brightness.dark
                                ? 'assets/images/Colorcard-hex_14181B-www.colorbook.io.png'
                                : 'assets/images/f1f4f8.png',
                            width: 200.0,
                            height: 200.0,
                            fit: BoxFit.cover,
                          ),
                        ),
                        FFButtonWidget(
                          onPressed: () async {
                            await actions.sendBackwards(
                              BTDeviceStruct(
                                name: widget!.deviceName,
                                id: widget!.deviceId,
                                rssi: widget!.deviceRssi,
                              ),
                              valueOrDefault<String>(
                                '',
                                '2',
                              ),
                            );
                          },
                          text: 'Backward',
                          options: FFButtonOptions(
                            width: 380.0,
                            height: 60.0,
                            padding: EdgeInsetsDirectional.fromSTEB(
                                16.0, 0.0, 16.0, 0.0),
                            iconPadding: EdgeInsetsDirectional.fromSTEB(
                                0.0, 0.0, 0.0, 0.0),
                            color: FlutterFlowTheme.of(context).primary,
                            textStyle: FlutterFlowTheme.of(context)
                                .titleSmall
                                .override(
                                  fontFamily: 'Montserrat',
                                  color: Colors.white,
                                  letterSpacing: 0.0,
                                ),
                            elevation: 0.0,
                            borderRadius: BorderRadius.circular(8.0),
                          ),
                        ),
                        ClipRRect(
                          borderRadius: BorderRadius.circular(8.0),
                          child: Image.asset(
                            Theme.of(context).brightness == Brightness.dark
                                ? 'assets/images/Colorcard-hex_14181B-www.colorbook.io.png'
                                : 'assets/images/f1f4f8.png',
                            width: 200.0,
                            height: 200.0,
                            fit: BoxFit.cover,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
                Padding(
                  padding: EdgeInsetsDirectional.fromSTEB(30.0, 24.0, 0.0, 0.0),
                  child: FFButtonWidget(
                    onPressed: () async {
                      await actions.sendStop(
                        BTDeviceStruct(
                          name: widget!.deviceName,
                          id: widget!.deviceId,
                          rssi: widget!.deviceRssi,
                        ),
                        valueOrDefault<String>(
                          '',
                          '0',
                        ),
                      );
                    },
                    text: 'Stop',
                    options: FFButtonOptions(
                      width: 320.0,
                      height: 80.0,
                      padding:
                          EdgeInsetsDirectional.fromSTEB(16.0, 0.0, 16.0, 0.0),
                      iconPadding:
                          EdgeInsetsDirectional.fromSTEB(0.0, 0.0, 0.0, 0.0),
                      color: FlutterFlowTheme.of(context).primary,
                      textStyle:
                          FlutterFlowTheme.of(context).titleSmall.override(
                                fontFamily: 'Montserrat',
                                color: Colors.white,
                                fontSize: 32.0,
                                letterSpacing: 0.0,
                              ),
                      elevation: 0.0,
                      borderRadius: BorderRadius.circular(8.0),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
