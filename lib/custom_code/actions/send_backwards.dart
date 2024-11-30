// Automatic FlutterFlow imports
import '/backend/backend.dart';
import '/backend/schema/structs/index.dart';
import '/flutter_flow/flutter_flow_theme.dart';
import '/flutter_flow/flutter_flow_util.dart';
import 'index.dart'; // Imports other custom actions
import 'package:flutter/material.dart';
// Begin custom action code
// DO NOT REMOVE OR MODIFY THE CODE ABOVE!

import 'index.dart'; // Imports other custom actions

import 'package:flutter_blue_plus/flutter_blue_plus.dart';

Future sendBackwards(BTDeviceStruct deviceInfo, String data) async {
  try {
    final device = BluetoothDevice.fromId(deviceInfo.id);
    await device.connect();
    debugPrint("Device connected: ${device.name}");

    final services = await device.discoverServices();
    for (BluetoothService service in services) {
      for (BluetoothCharacteristic characteristic in service.characteristics) {
        debugPrint("Found characteristic: ${characteristic.uuid.toString()}");

        // Check if this is the characteristic you want to write to
        if (characteristic.uuid.toString() ==
                '0000bbbb-beef-c0c0-c0de-c0ffeefacade' &&
            characteristic.properties.write) {
          debugPrint(
              "Found writable characteristic: ${characteristic.uuid.toString()}");

          // Send each character as a single byte to the characteristic
          for (int i = 0; i < data.length; i++) {
            String charToSend = data[i];
            debugPrint("Sending character: $charToSend");
            await characteristic.write(charToSend.codeUnits,
                withoutResponse: false); // Set to false to get confirmation
            await Future.delayed(Duration(
                milliseconds: 100)); // Small delay to prevent rapid sending
          }

          debugPrint("Successfully wrote all characters.");
          return; // Exit after successful write
        }
      }
    }

    debugPrint("No writable characteristic found.");
  } catch (e) {
    debugPrint("Error in sendData: ${e.toString()}");
  }
}
