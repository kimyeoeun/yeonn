//
//  detection_app_2App.swift
//  detection_app_2
//
//  Created by 김여은 on 4/25/24.
//

import SwiftUI
import AVFoundation


@main
struct detection_app_2App: App {
    @StateObject private var userManager = UserManager()
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(userManager)
        }
    }
}





