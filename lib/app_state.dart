import 'package:flutter/material.dart';
import '/backend/backend.dart';
import '/backend/schema/structs/index.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'flutter_flow/flutter_flow_util.dart';

class FFAppState extends ChangeNotifier {
  static FFAppState _instance = FFAppState._internal();

  factory FFAppState() {
    return _instance;
  }

  FFAppState._internal();

  static void reset() {
    _instance = FFAppState._internal();
  }

  Future initializePersistedState() async {}

  void update(VoidCallback callback) {
    callback();
    notifyListeners();
  }

  List<String> _directions = [];
  List<String> get directions => _directions;
  set directions(List<String> value) {
    _directions = value;
  }

  void addToDirections(String value) {
    directions.add(value);
  }

  void removeFromDirections(String value) {
    directions.remove(value);
  }

  void removeAtIndexFromDirections(int index) {
    directions.removeAt(index);
  }

  void updateDirectionsAtIndex(
    int index,
    String Function(String) updateFn,
  ) {
    directions[index] = updateFn(_directions[index]);
  }

  void insertAtIndexInDirections(int index, String value) {
    directions.insert(index, value);
  }
}
