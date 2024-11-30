import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/foundation.dart';

Future initFirebase() async {
  if (kIsWeb) {
    await Firebase.initializeApp(
        options: FirebaseOptions(
            apiKey: "AIzaSyCg9cBYqeJTG-lUdzk8595lCcdG0Unh5RM",
            authDomain: "newtest-228c3.firebaseapp.com",
            projectId: "newtest-228c3",
            storageBucket: "newtest-228c3.appspot.com",
            messagingSenderId: "308547041733",
            appId: "1:308547041733:web:3ff748b9a6948ba136f3ac",
            measurementId: "G-41XWP7828G"));
  } else {
    await Firebase.initializeApp();
  }
}
