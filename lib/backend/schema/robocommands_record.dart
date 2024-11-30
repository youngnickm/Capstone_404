import 'dart:async';

import 'package:collection/collection.dart';

import '/backend/schema/util/firestore_util.dart';
import '/backend/schema/util/schema_util.dart';

import 'index.dart';
import '/flutter_flow/flutter_flow_util.dart';

class RobocommandsRecord extends FirestoreRecord {
  RobocommandsRecord._(
    DocumentReference reference,
    Map<String, dynamic> data,
  ) : super(reference, data) {
    _initializeFields();
  }

  // "directions" field.
  String? _directions;
  String get directions => _directions ?? '';
  bool hasDirections() => _directions != null;

  void _initializeFields() {
    _directions = snapshotData['directions'] as String?;
  }

  static CollectionReference get collection =>
      FirebaseFirestore.instance.collection('robocommands');

  static Stream<RobocommandsRecord> getDocument(DocumentReference ref) =>
      ref.snapshots().map((s) => RobocommandsRecord.fromSnapshot(s));

  static Future<RobocommandsRecord> getDocumentOnce(DocumentReference ref) =>
      ref.get().then((s) => RobocommandsRecord.fromSnapshot(s));

  static RobocommandsRecord fromSnapshot(DocumentSnapshot snapshot) =>
      RobocommandsRecord._(
        snapshot.reference,
        mapFromFirestore(snapshot.data() as Map<String, dynamic>),
      );

  static RobocommandsRecord getDocumentFromData(
    Map<String, dynamic> data,
    DocumentReference reference,
  ) =>
      RobocommandsRecord._(reference, mapFromFirestore(data));

  @override
  String toString() =>
      'RobocommandsRecord(reference: ${reference.path}, data: $snapshotData)';

  @override
  int get hashCode => reference.path.hashCode;

  @override
  bool operator ==(other) =>
      other is RobocommandsRecord &&
      reference.path.hashCode == other.reference.path.hashCode;
}

Map<String, dynamic> createRobocommandsRecordData({
  String? directions,
}) {
  final firestoreData = mapToFirestore(
    <String, dynamic>{
      'directions': directions,
    }.withoutNulls,
  );

  return firestoreData;
}

class RobocommandsRecordDocumentEquality
    implements Equality<RobocommandsRecord> {
  const RobocommandsRecordDocumentEquality();

  @override
  bool equals(RobocommandsRecord? e1, RobocommandsRecord? e2) {
    return e1?.directions == e2?.directions;
  }

  @override
  int hash(RobocommandsRecord? e) => const ListEquality().hash([e?.directions]);

  @override
  bool isValidKey(Object? o) => o is RobocommandsRecord;
}
