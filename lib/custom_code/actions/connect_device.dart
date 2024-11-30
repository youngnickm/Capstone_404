// Automatic FlutterFlow imports
import '/backend/backend.dart';
import '/backend/schema/structs/index.dart';
import '/flutter_flow/flutter_flow_theme.dart';
import '/flutter_flow/flutter_flow_util.dart';
import 'index.dart'; // Imports other custom actions
import 'package:flutter/material.dart';
// Begin custom action code
// DO NOT REMOVE OR MODIFY THE CODE ABOVE!

import 'package:flutter_blue_plus/flutter_blue_plus.dart'; // Importing FlutterBluePlus library for BT

// Function for connecting to a BT device
Future<bool> connectDevice(BTDeviceStruct deviceInfo) async {
  final device = BluetoothDevice.fromId(
      deviceInfo.id); // BT device instance creation using deviceInfo
  // Flag to check if the device has write characteristic in later search
  var hasWriteCharacteristic = false;
  // Attempting to connect to a device, catching errors in the catch block
  try {
    await device.connect();
    // Iterating through each device's characteristics and checking if there's a characteristic that supports write
    // Returns true if the device does, false if not
    // Searching for 'services' or devices with BT
    final services = await device.discoverServices();
    for (BluetoothService service in services) {
      for (BluetoothCharacteristic characteristic in service.characteristics) {
        final isWrite = characteristic.properties.write;
        if (isWrite) {
          debugPrint(
              'Found write characteristic: ${characteristic.uuid}, ${characteristic.properties}');
          hasWriteCharacteristic = true;
        }
      }
    }
  } catch (e) {
    debugPrint(e.toString());
  }
  return hasWriteCharacteristic;
}
