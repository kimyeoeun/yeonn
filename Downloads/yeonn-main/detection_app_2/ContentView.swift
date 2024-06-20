import UIKit
import AVFoundation
import Vision


struct ObservationData: Codable {
    let boundingBox: CGRect
    let confidence: VNConfidence
    let identifier: String
}

extension VNRecognizedObjectObservation {
    var observationData: ObservationData {
        return ObservationData(
            boundingBox: self.boundingBox,
            confidence: self.confidence,
            identifier: self.labels.first?.identifier ?? "Unknown"
        )
    }
}



class CameraViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    var bufferSize: CGSize = .zero
    var inferenceTime: CFTimeInterval = 0
    private let session = AVCaptureSession()
    private var previewLayer: AVCaptureVideoPreviewLayer!
    private var detectionLayer: CALayer! = nil
    private var requests = [VNRequest]()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupCapture()
        setupOutput()
        setupLayers()
        try? setupVision()
        session.startRunning()
    }
    
    func setupCapture() {
        guard let videoDevice = AVCaptureDevice.default(for: .video) else {
            fatalError("No video device available")
        }
        do {
            let deviceInput = try AVCaptureDeviceInput(device: videoDevice)
            session.beginConfiguration()
            if session.canAddInput(deviceInput) {
                session.addInput(deviceInput)
            } else {
                fatalError("Could not add video device input to the session")
            }
            session.sessionPreset = .vga640x480
            let dimensions = CMVideoFormatDescriptionGetDimensions(videoDevice.activeFormat.formatDescription)
            bufferSize.width = CGFloat(dimensions.width)
            bufferSize.height = CGFloat(dimensions.height)
            try videoDevice.lockForConfiguration()
            videoDevice.unlockForConfiguration()
            session.commitConfiguration()
        } catch {
            fatalError("Error configuring the video device: \(error)")
        }
    }
    
    func setupOutput() {
        let videoDataOutput = AVCaptureVideoDataOutput()
        let videoDataOutputQueue = DispatchQueue(label: "VideoDataOutput", qos: .userInitiated, attributes: [], autoreleaseFrequency: .workItem)
        if session.canAddOutput(videoDataOutput) {
            session.addOutput(videoDataOutput)
            videoDataOutput.alwaysDiscardsLateVideoFrames = true
            videoDataOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)]
            videoDataOutput.setSampleBufferDelegate(self, queue: videoDataOutputQueue)
        } else {
            fatalError("Could not add video data output to the session")
        }
    }
    
    func setupLayers() {
        previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.frame = view.layer.bounds
        view.layer.addSublayer(previewLayer)
        
        detectionLayer = CALayer()
        detectionLayer.bounds = CGRect(x: 0.0, y: 0.0, width: bufferSize.width, height: bufferSize.height)
        detectionLayer.position = CGPoint(x: view.bounds.midX, y: view.bounds.midY)
        view.layer.addSublayer(detectionLayer)
        
        let xScale: CGFloat = view.bounds.size.width / bufferSize.height
        let yScale: CGFloat = view.bounds.size.height / bufferSize.width
        let scale = max(xScale, yScale)
        detectionLayer.setAffineTransform(CGAffineTransform(rotationAngle: .pi / 2).scaledBy(x: scale, y: -scale))
        detectionLayer.position = CGPoint(x: view.bounds.midX, y: view.bounds.midY)
    }
    
    func setupVision() throws {
        guard let modelURL = Bundle.main.url(forResource: "yolov5n", withExtension: "mlmodelc") else {
            throw NSError(domain: "ViewController", code: -1, userInfo: [NSLocalizedDescriptionKey: "Model file is missing"])
        }
        do {
            let visionModel = try VNCoreMLModel(for: MLModel(contentsOf: modelURL))
            let objectRecognition = VNCoreMLRequest(model: visionModel, completionHandler: { (request, error) in
                DispatchQueue.main.async {
                    if let results = request.results {
                        self.drawResults(results)
                    }
                }
            })
            self.requests = [objectRecognition]
        } catch let error as NSError {
            print("Model loading went wrong: \(error)")
        }
    }
    
    func drawResults(_ results: [Any]) {
        var detectedObjects: [VNRecognizedObjectObservation] = []
        
        CATransaction.begin()
        CATransaction.setValue(kCFBooleanTrue, forKey: kCATransactionDisableActions)
        detectionLayer.sublayers?.removeAll()
        
        for result in results {
            if let observation = result as? VNRecognizedObjectObservation {
                detectedObjects.append(observation)
                let objectBounds = VNImageRectForNormalizedRect(observation.boundingBox, Int(bufferSize.width), Int(bufferSize.height))
                let shapeLayer = createRectLayer(objectBounds, [0, 0, 1, 0.5])
                
                let topLabel = observation.labels.first?.identifier ?? "Unknown"
                let confidence = observation.labels.first?.confidence ?? 0
                let labelString = "\(topLabel) \(Int(confidence * 100))%"
                let textLayer = createTextLayer(label: labelString, frame: objectBounds)
                shapeLayer.addSublayer(textLayer)
                
                detectionLayer.addSublayer(shapeLayer)
            }
        }
        
        // 커뮤니티 사진들과 비교하는 함수 호출
        compareWithCommunityImages(detectedObjects)
        
        CATransaction.commit()
    }
    
    func createRectLayer(_ frame: CGRect, _ colorComponents: [CGFloat]) -> CALayer {
        let layer = CALayer()
        layer.frame = frame
        layer.backgroundColor = CGColor(red: colorComponents[0], green: colorComponents[1], blue: colorComponents[2], alpha: colorComponents[3])
        return layer
    }
    
    func createTextLayer(label: String, frame: CGRect) -> CATextLayer {
        let textLayer = CATextLayer()
        textLayer.string = label
        textLayer.fontSize = 10
        textLayer.alignmentMode = .center
        textLayer.foregroundColor = CGColor(red: 1, green: 1, blue: 1, alpha: 1)
        textLayer.frame = frame
        textLayer.contentsScale = UIScreen.main.scale
        return textLayer
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        
        let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right, options: [:])
        do {
            let start = CACurrentMediaTime()
            try imageRequestHandler.perform(self.requests)
            inferenceTime = (CACurrentMediaTime() - start)
        } catch {
            print(error)
        }
    }
    
    // 커뮤니티 사진과 비교하는 함수들 추가
    func compareWithCommunityImages(_ detectedObjects: [VNRecognizedObjectObservation]) {
        let communityPosts = loadCommunityPosts()
        var highestSimilarity: Float = 0
        var mostSimilarPost: Post? = nil
        
        for post in communityPosts {
            if let postImage = post.image {
                let postObservations = detectObjectsInImage(postImage)
                let similarity = calculateSimilarity(detectedObjects, postObservations)
                
                if similarity > highestSimilarity {
                    highestSimilarity = similarity
                    mostSimilarPost = post
                }
            }
        }
        
        if let mostSimilarPost = mostSimilarPost {
            showMostSimilarPost(mostSimilarPost)
        }
    }
    
    func detectObjectsInImage(_ image: UIImage) -> [VNRecognizedObjectObservation] {
        guard let ciImage = CIImage(image: image) else {
            return []
        }
        
        do {
            let model = try VNCoreMLModel(for: MLModel(contentsOf: Bundle.main.url(forResource: "yolov5n", withExtension: "mlmodelc")!))
            let request = VNCoreMLRequest(model: model, completionHandler: { (request, error) in
                // Handle error if needed
            })
            
            let requestHandler = VNImageRequestHandler(ciImage: ciImage, options: [:])
            
            try requestHandler.perform([request])
            return request.results as? [VNRecognizedObjectObservation] ?? []
        } catch {
            print("Failed to perform request: \(error)")
            return []
        }
    }
    
    
    func calculateSimilarity(_ detectedObjects: [VNRecognizedObjectObservation], _ postObservations: [VNRecognizedObjectObservation]) -> Float {
        // 두 객체 목록 간의 유사도를 계산하는 로직 (예제: 단순히 객체 수를 비교)
        if detectedObjects.isEmpty || postObservations.isEmpty {
            return 0.0
        }
        
        let detectedSet = Set(detectedObjects.map { $0.labels.first?.identifier ?? "Unknown" })
        let postSet = Set(postObservations.map { $0.labels.first?.identifier ?? "Unknown" })
        
        let intersection = detectedSet.intersection(postSet)
        let union = detectedSet.union(postSet)
        
        return Float(intersection.count) / Float(union.count)
    }
    
    func loadCommunityPosts() -> [Post] {
        if let savedData = UserDefaults.standard.data(forKey: "posts"),
           let decodedPosts = try? JSONDecoder().decode([Post].self, from: savedData) {
            return decodedPosts
        }
        return []
    }
    
    func showMostSimilarPost(_ post: Post) {
        if let image = post.image {
            let alert = UIAlertController(title: "가장 유사한 사진", message: nil, preferredStyle: .alert)
            let imageView = UIImageView(image: image)
            imageView.contentMode = .scaleAspectFit
            alert.view.addSubview(imageView)
            
            imageView.translatesAutoresizingMaskIntoConstraints = false
            NSLayoutConstraint.activate([
                imageView.widthAnchor.constraint(equalToConstant: 250),
                imageView.heightAnchor.constraint(equalToConstant: 250),
                imageView.centerXAnchor.constraint(equalTo: alert.view.centerXAnchor),
                imageView.topAnchor.constraint(equalTo: alert.view.topAnchor, constant: 50)
            ])
            
            let okAction = UIAlertAction(title: "확인", style: .default, handler: nil)
            alert.addAction(okAction)
            
            present(alert, animated: true, completion: nil)
        }
    }
}

import SwiftUI
import UIKit
import AVFoundation


struct ContentView: View {
    @State private var cameraImage: UIImage?
    @State private var showMainView = true
    @State private var logoScale: CGFloat = 0.8
    @State private var logoOpacity: Double = 0.0
    @State private var isLoggedIn: Bool = false
    @State private var posts: [Post] = []

    var body: some View {
        NavigationView {
            ZStack {
                if showMainView || !showMainView && !isLoggedIn || isLoggedIn {
                    Color.yellow
                        .edgesIgnoringSafeArea(.all)
                }

                VStack {
                    if showMainView {
                        Spacer()
                        Image("logo")
                            .resizable()
                            .scaledToFit()
                            .frame(width: 200, height: 200)
                            .scaleEffect(logoScale)
                            .opacity(logoOpacity)
                            .onAppear {
                                withAnimation(.easeIn(duration: 1.5)) {
                                    self.logoScale = 1.0
                                    self.logoOpacity = 1.0
                                }
                            }

                        Text("찾았Dog")
                            .font(.largeTitle)
                            .fontWeight(.bold)
                            .foregroundColor(.black)
                            .padding()
                            .opacity(logoOpacity)
                            .onAppear {
                                withAnimation(.easeIn(duration: 1.5).delay(0.5)) {
                                    self.logoOpacity = 1.0
                                }
                            }

                        Spacer()
                    } else if isLoggedIn {
                        VStack {
                            Spacer().frame(height: 50)
                            
                            Text("찾았Dog")
                                .font(.largeTitle)
                                .fontWeight(.bold)
                                .padding()

                            Spacer()

                            HStack(spacing: 0) { // 간격을 최소화했습니다.
                                IconView(imageName: "logo2", title: "애완견", subtitle: "등록하기", destination: AnyView(CameraViewControllerRepresentable(image: $cameraImage)))
                                IconView(imageName: "logo2", title: "커뮤니티", subtitle: "입장하기", destination: AnyView(CommunityView(posts: $posts)))
                            }
                            .padding(.horizontal, 10)
                            .padding(.vertical, 10) // 세로 간격을 줄였습니다.

                            Spacer()
                        }
                        .navigationBarHidden(true)
                    } else {
                        LoginViewControllerRepresentable(isLoggedIn: $isLoggedIn)
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(
                    Color.yellow
                        .edgesIgnoringSafeArea(.all)
                )
            }
            .onAppear {
                if showMainView {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
                        withAnimation {
                            showMainView = false
                        }
                    }
                }
            }
        }
    }
}


struct IconView: View {
    var imageName: String
    var title: String
    var subtitle: String
    var destination: AnyView

    var body: some View {
        NavigationLink(destination: destination) {
            VStack(spacing: 5) { // 내부 간격을 조정했습니다.
                Image(imageName)
                    .resizable()
                    .scaledToFit()
                    .frame(width: 100, height: 100)
                
                VStack(alignment: .center, spacing: 5) { // 내부 간격을 조정했습니다.
                    Text(title)
                        .font(.title)
                        .fontWeight(.bold)
                        .foregroundColor(.black)
                    Text(subtitle)
                        .font(.headline)
                        .foregroundColor(.gray)
                }
                .padding(.horizontal, 10) // 좌우 패딩을 줄였습니다.
                .background(Color.white)
                .cornerRadius(15)
                .shadow(radius: 10)
                .frame(width: 160, height: 120) // 전체 크기를 줄였습니다.
            }
            .padding(.horizontal, 10) // 외부 패딩을 줄였습니다.
        }
    }
}





struct CameraViewControllerRepresentable: UIViewControllerRepresentable {
    class Coordinator: NSObject, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
        var parent: CameraViewControllerRepresentable
        
        init(parent: CameraViewControllerRepresentable) {
            self.parent = parent
        }
        
        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            if let image = info[.originalImage] as? UIImage {
                parent.image = image
            }
            parent.presentationMode.wrappedValue.dismiss()
        }
        
        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            parent.presentationMode.wrappedValue.dismiss()
        }
    }
    
    @Environment(\.presentationMode) var presentationMode
    @Binding var image: UIImage?
    
    func makeCoordinator() -> Coordinator {
        Coordinator(parent: self)
    }
    
    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.delegate = context.coordinator
        picker.sourceType = .camera
        return picker
    }
    
    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}
    
}

struct GalleryViewControllerRepresentable: UIViewControllerRepresentable {
    class Coordinator: NSObject, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
        var parent: GalleryViewControllerRepresentable
        
        init(parent: GalleryViewControllerRepresentable) {
            self.parent = parent
        }
        
        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            // 선택한 이미지를 처리합니다.
            if let image = info[.originalImage] as? UIImage {
                // 여기서 이미지를 사용합니다.
            }
            parent.presentationMode.wrappedValue.dismiss()
        }
        
        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            parent.presentationMode.wrappedValue.dismiss()
        }
    }
    
    @Environment(\.presentationMode) var presentationMode
    
    func makeCoordinator() -> Coordinator {
        Coordinator(parent: self)
    }
    
    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.delegate = context.coordinator
        picker.sourceType = .photoLibrary
        return picker
    }
    
    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}
}



struct CommunityView: View {
    @State private var text: String = ""
    @State private var image: UIImage?
    @State private var isImagePickerPresented = false
    @State private var isCameraPickerPresented = false
    @Binding var posts: [Post]
    @State private var isFeedViewActive = false
    @State private var editingPost: Post? = nil
    @EnvironmentObject var userManager: UserManager
    @State private var navigateToFeedView = false  // 상태 변수 선언
    private let postsKey = "posts"
    
    init(posts: Binding<[Post]>, editingPost: Post? = nil) {
        self._posts = posts
        self._editingPost = State(initialValue: editingPost)
    }
    
    var body: some View {
        VStack {
            Text("커뮤니티")
                .font(.largeTitle)
                .fontWeight(.bold)
                .padding()
            
            if let image = image {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(height: 200)
            } else {
                HStack {
                    Button(action: {
                        isImagePickerPresented = true
                    }) {
                        Text("사진 추가")
                            .font(.title)
                            .padding()
                            .background(Color.blue)
                            .foregroundColor(.white)
                            .cornerRadius(10)
                    }
                    .sheet(isPresented: $isImagePickerPresented) {
                        ImagePicker(image: $image)
                    }
                    
                    Button(action: {
                        isCameraPickerPresented = true
                    }) {
                        Text("사진 촬영")
                            .font(.title)
                            .padding()
                            .background(Color.green)
                            .foregroundColor(.white)
                            .cornerRadius(10)
                    }
                    .sheet(isPresented: $isCameraPickerPresented) {
                        CameraViewControllerRepresentable(image: $image)
                    }
                }
            }
            
            TextField("내용을 입력하세요...", text: $text)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .padding()
            
            Button(action: {
                if let editingPost = editingPost {
                    updateContent(editingPost)
                } else {
                    uploadContent()
                }
                navigateToFeedView = true
            }) {
                Text(editingPost == nil ? "확인" : "업데이트")
                    .font(.title)
                    .padding()
                    .background(Color.orange)
                    .foregroundColor(.white)
                    .cornerRadius(10)
            }
            .padding(.top)
            
            Spacer()
            
            NavigationLink(destination: FeedView(posts: $posts)) {
                Text("피드로 가기")
                
                    .font(.title)
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
            }
            .padding(.top)
            
            Spacer()
        }
        .padding()
        .onAppear {
            loadPosts()
            if let editingPost = editingPost {
                text = editingPost.text
                if let imageData = editingPost.imageData {
                    image = UIImage(data: imageData)
                }
                NavigationLink(destination: FeedView(posts: $posts), isActive: $navigateToFeedView) {
                    EmptyView()
                }
            }
        }
    }
    
    func uploadContent() {
        guard !text.isEmpty || image != nil else { return }
        let newPost = Post(id: UUID(), text: text, imageData: image?.jpegData(compressionQuality: 0.8), author: userManager.currentUser, observations: [])
        posts.append(newPost)
        savePosts()
        resetForm()
    }
    
    func updateContent(_ post: Post) {
        if let index = posts.firstIndex(where: { $0.id == post.id }) {
            posts[index].text = text
            posts[index].imageData = image?.jpegData(compressionQuality: 0.8)
            savePosts()
            resetForm()
        }
    }
    
    func resetForm() {
        text = ""
        image = nil
        editingPost = nil
        isFeedViewActive = true
    }
    
    func savePosts() {
        if let encodedData = try? JSONEncoder().encode(posts) {
            UserDefaults.standard.set(encodedData, forKey: postsKey)
        }
    }
    
    func loadPosts() {
        if let savedData = UserDefaults.standard.data(forKey: postsKey),
           let decodedPosts = try? JSONDecoder().decode([Post].self, from: savedData) {
            posts = decodedPosts
        }
    }
}




struct FeedView: View {
    @Binding var posts: [Post]
    @EnvironmentObject var userManager: UserManager
    @State private var showCommunityView = false
    @State private var postToEdit: Post? = nil
    
    var body: some View {
        List {
            ForEach(posts) { post in
                VStack(alignment: .leading) {
                    if let postImage = post.image {
                        Image(uiImage: postImage)
                            .resizable()
                            .scaledToFit()
                            .frame(height: 200)
                    }
                    Text(post.text)
                        .padding(.top, 5)
                    
                    if post.author == userManager.currentUser {
                        HStack {
                            Button("수정") {
                                postToEdit = post
                                showCommunityView = true
                            }
                            .foregroundColor(.blue)
                            .padding(.trailing)
                            
                            Button("삭제") {
                                deletePost(post)
                                //showCommunityView = false
                                
                            }
                            .foregroundColor(.red)
                        }
                        .padding(.top, 5)
                    }
                }
                .padding()
            }
            .onDelete { indexSet in
                indexSet.forEach { index in
                    let post = posts[index]
                    if post.author == userManager.currentUser {
                        posts.remove(at: index)
                        savePosts()
                    }
                }
            }
        }
        .navigationTitle("피드")
        .sheet(isPresented: $showCommunityView) {
            if let postToEdit = postToEdit {
                CommunityView(posts: $posts, editingPost: postToEdit)
                    .environmentObject(userManager) // EnvironmentObject 전달
            }
        }
    }
    
    func deletePost(_ post: Post) {
        if let index = posts.firstIndex(where: { $0.id == post.id }) {
            posts.remove(at: index)
            savePosts()
        }
    }
    
    func savePosts() {
        if let encodedData = try? JSONEncoder().encode(posts) {
            UserDefaults.standard.set(encodedData, forKey: "posts")
        }
    }
}







struct Post: Identifiable, Codable {
    let id: UUID
    var text: String // 변경 가능하도록 var로 수정
    var imageData: Data? // 변경 가능하도록 var로 수정
    let author: String
    let observations: [ObservationData]?
    
    var image: UIImage? {
        if let imageData = imageData {
            return UIImage(data: imageData)
        }
        return nil
    }
}


class UserManager: ObservableObject {
    @Published var currentUser: String = ""
}



struct ImagePicker: UIViewControllerRepresentable {
    @Binding var image: UIImage?
    
    class Coordinator: NSObject, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
        let parent: ImagePicker
        
        init(parent: ImagePicker) {
            self.parent = parent
        }
        
        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            if let uiImage = info[.originalImage] as? UIImage {
                parent.image = uiImage
            }
            picker.dismiss(animated: true)
        }
        
        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            picker.dismiss(animated: true)
        }
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator(parent: self)
    }
    
    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.delegate = context.coordinator
        picker.sourceType = .photoLibrary
        return picker
    }
    
    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}
}


struct User: Codable {
    let username: String
    let password: String
}


struct LoginViewControllerRepresentable: View {
    @State private var username: String = ""
    @State private var password: String = ""
    @State private var showSignUp: Bool = false
    @State private var loginErrorMessage: String? = nil
    @Binding var isLoggedIn: Bool
    @EnvironmentObject var userManager: UserManager

    var body: some View {
        ZStack {
            Image("Rectangle")
                .resizable()
                .scaledToFill()
                .edgesIgnoringSafeArea(.all)

            VStack {
                Spacer()

                // 커스텀 타이틀
                Text("로그인")
                    .font(.system(size: 40, weight: .bold)) // 원하는 글꼴 크기와 스타일 지정
                    .foregroundColor(.black)
                    .padding(.bottom, 20)

                // 로고 이미지 추가
                Image("logo")
                    .resizable()
                    .scaledToFit()
                    .frame(width: 100, height: 100)
                    .padding(.bottom, 20)

                // 아이디 입력 필드
                TextField("아이디", text: $username)
                    .padding()
                    .background(Color.white)
                    .cornerRadius(10)
                    .padding([.leading, .trailing], 20)

                // 비밀번호 입력 필드
                SecureField("비밀번호", text: $password)
                    .padding()
                    .background(Color.white)
                    .cornerRadius(10)
                    .padding([.leading, .trailing], 20)

                // 로그인 버튼
                Button(action: {
                    login()
                }) {
                    Text("로그인")
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.white)
                        .foregroundColor(.black)
                        .cornerRadius(10)
                        .padding([.leading, .trailing], 20)
                }
                .padding()

                if let message = loginErrorMessage {
                    Text(message)
                        .foregroundColor(.red)
                        .padding()
                }

                // 회원가입 버튼
                Button(action: {
                    showSignUp = true
                }) {
                    Text("회원가입")
                        .font(.footnote)
                        .underline()
                }
                .padding(.top, 20)

                Spacer()
            }
            .padding()
            .sheet(isPresented: $showSignUp) {
                SignUpView(showSignUp: $showSignUp, isLoggedIn: $isLoggedIn)
            }
        }
    }

    private func login() {
        if let userData = UserDefaults.standard.data(forKey: username),
           let savedUser = try? JSONDecoder().decode(User.self, from: userData),
           savedUser.password == password {
            userManager.currentUser = username
            isLoggedIn = true
        } else {
            loginErrorMessage = "아이디 또는 비밀번호가 잘못되었습니다."
        }
    }
}





struct SignUpView: View {
    @Binding var showSignUp: Bool
    @Binding var isLoggedIn: Bool
    @State private var newUsername: String = ""
    @State private var newPassword: String = ""
    @State private var confirmPassword: String = ""
    @State private var signUpErrorMessage: String? = nil
    @EnvironmentObject var userManager: UserManager
    
    var body: some View {
        VStack {
            TextField("새 아이디", text: $newUsername)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .padding()
            
            SecureField("새 비밀번호", text: $newPassword)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .padding()
            
            SecureField("비밀번호 확인", text: $confirmPassword)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .padding()
            
            Button(action: {
                signUp()
            }) {
                Text("회원가입")
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.green)
                    .foregroundColor(.white)
                    .cornerRadius(10)
            }
            .padding()
            
            if let message = signUpErrorMessage {
                Text(message)
                    .foregroundColor(.red)
                    .padding()
            }
            
            Spacer()
        }
        .padding()
        .navigationBarTitle("회원가입")
    }
    
    private func signUp() {
        guard newPassword == confirmPassword else {
            signUpErrorMessage = "비밀번호가 일치하지 않습니다."
            return
        }
        
        let newUser = User(username: newUsername, password: newPassword)
        if let encodedData = try? JSONEncoder().encode(newUser) {
            UserDefaults.standard.set(encodedData, forKey: newUsername)
            userManager.currentUser = newUsername
            showSignUp = false
            isLoggedIn = true
        } else {
            signUpErrorMessage = "회원가입 중 오류가 발생했습니다."
        }
    }
}




struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
            .environmentObject(UserManager())
    }
}

