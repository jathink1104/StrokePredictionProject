import os
from flask import Flask, session, render_template, request, redirect, url_for, jsonify, send_from_directory
import tensorflow as tf
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import pyodbc
import matplotlib.pyplot as plt
from werkzeug.security import generate_password_hash
# Tắt tối ưu hóa oneDNN của TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Khởi tạo Flask app
app = Flask(__name__, template_folder='frontend/ui')

# Cấu hình thư mục tải lên ảnh
UPLOAD_FOLDER = 'backend/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['SECRET_KEY'] = 's3cr3t_k3y_123456'  # Thay thế bằng chuỗi bạn đã tạo  

# Phục vụ các tệp tĩnh (CSS, JS, hình ảnh) từ frontend
@app.route('/frontend/<path:path>')
def static_files(path): 
    return send_from_directory('frontend', path)


# Phục vụ các tệp tĩnh backend upload
@app.route('/backend/<path:path>')
def static_backend_files(path):
    return send_from_directory('backend', path)
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # Trả về hình ảnh từ thư mục uploads
    return send_from_directory(UPLOAD_FOLDER, filename)

# Kiểm tra file có hợp lệ không
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Kết nối đến SQL Server
def connect_to_sql_server():
    server = r'MSI\SQLEXPRESS'  
    database = 'StrokePrediction'  
    username = ''  
    password = '' 

    try:
        conn = pyodbc.connect(
            f'DRIVER={{SQL Server}};'
            f'SERVER={server};'
            f'DATABASE={database};'
            f'UID={username};'
            f'PWD={password};'
        )
        print("Kết nối cơ sở dữ liệu thành công!")
        return conn
    except Exception as e:
        print(f"Error connecting to SQL Server: {e}")
        return None

# Kiểm tra đăng nhập của người dùng
def check_user_login(username, password):
    try:
        conn = connect_to_sql_server()
        if conn:
            cursor = conn.cursor()

            # Truy vấn cơ sở dữ liệu để kiểm tra tài khoản
            cursor.execute("SELECT Id, username, password FROM userAcount WHERE username=?", (username,))
            user = cursor.fetchone()

            cursor.close()
            conn.close()

            # Nếu tìm thấy người dùng thì so sánh mật khẩu
            if user and user[2] == password:  # So sánh mật khẩu rõ ràng
                return user
            else:
                return None
        else:
            return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

# Route đăng nhập
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Kiểm tra thông tin đăng nhập từ cơ sở dữ liệu
        user = check_user_login(username, password)

        if user:
            # Đăng nhập thành công, lưu thông tin vào session
            session['user_id'] = user[0]  # Lưu ID người dùng vào session
            session['username'] = user[1]  # Lưu username vào session
            return redirect(url_for('index'))  # Chuyển hướng đến trang index sau khi đăng nhập
        else:
            # Nếu thông tin đăng nhập sai
            return render_template('login.html', error="Invalid username or password")

    return render_template('login.html')

# Route đăng ký
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        gmail = request.form['email']
        address = request.form.get('address', '')  # Có thể để trống
        phonenumber = request.form.get('phonenumber', '')

        # Kiểm tra nếu người dùng đã tồn tại trong cơ sở dữ liệu
        conn = connect_to_sql_server()
        if conn:
            cursor = conn.cursor()

            # Kiểm tra xem username đã tồn tại chưa
            cursor.execute("SELECT COUNT(*) FROM userAcount WHERE username=?", (username,))
            if cursor.fetchone()[0] > 0:
                return render_template('register.html', error="Username already exists!")

            # Lưu mật khẩu ở dạng rõ ràng (plain text)
            cursor.execute("""
                INSERT INTO userAcount (username, password, gmail, address, phonenumber) 
                VALUES (?, ?, ?, ?, ?)
            """, (username, password, gmail, address, phonenumber))

            # Commit thay đổi vào cơ sở dữ liệu
            conn.commit()
            cursor.close()
            conn.close()

            return redirect(url_for('login'))  # Chuyển hướng người dùng về trang đăng nhập sau khi đăng ký thành công

    return render_template('register.html')

# Route kiểm tra kết nối cơ sở dữ liệu
@app.route('/check-db-connection')
def check_db_connection():
    conn = connect_to_sql_server()
    if conn:
        return 'Kết nối cơ sở dữ liệu thành công!'
    else:
        return 'Không thể kết nối cơ sở dữ liệu!'

# Hàm dự đoán ảnh
def predict_image(image_path):
    model = tf.keras.models.load_model('resnet50_final.h5')
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (224, 224))  # Resize ảnh về kích thước (224, 224)
    img_array = np.expand_dims(img_resized / 255.0, axis=0)  # Normalize và tạo batch

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)  # Lấy index của nhãn có xác suất cao nhất
    class_names = ['Haemorrhagic', 'Ischemic', 'Normal']
    predicted_class_label = class_names[predicted_class_index]

    return model, predicted_class_label, predictions[0][predicted_class_index]

# Hàm tính Grad-CAM
def compute_gradcam(model, image_path, layer_name):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img_resized / 255.0, axis=0)

    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        predicted_class = tf.argmax(predictions[0])
        loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)

    return heatmap.numpy()

# Hàm tạo ảnh Grad-CAM
def create_gradcam_image(image_path, heatmap, output_path, alpha=0.4):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

    plt.imsave(output_path, superimposed_img)

# Hàm tạo ảnh overlay với vùng tổn thương
def create_gradcam_with_overlay(image_path, heatmap, output_path, class_label):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    if class_label != "Normal":  # Chỉ vẽ đường viền nếu không phải "Normal"
        heatmap_uint8 = np.uint8(255 * heatmap)
        _, thresh = cv2.threshold(heatmap_uint8, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    plt.imsave(output_path, img)

# Route dự đoán
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    # Lấy thông tin bệnh nhân từ form
    patient_name = request.form['patient_name']
    email = request.form['email']
    phone_number = request.form['phone_number']

    # Xử lý ảnh và lưu vào thư mục
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Gọi hàm predict_image
        model, predicted_label, confidence = predict_image(filepath)

        # Chuyển đổi confidence từ numpy.float32 sang float
        confidence = float(confidence)

        # Tính Grad-CAM
        heatmap = compute_gradcam(
            model=model,
            image_path=filepath,
            layer_name='conv5_block3_out'
        )

        # Đường dẫn ảnh Grad-CAM và overlay
        gradcam_path = os.path.join(app.config['UPLOAD_FOLDER'], f"gradcam_{filename}")
        overlay_path = os.path.join(app.config['UPLOAD_FOLDER'], f"overlay_{filename}")

        create_gradcam_image(filepath, heatmap, gradcam_path)

        # Chỉ tạo overlay nếu không phải "Normal"
        if predicted_label != "Normal":
            create_gradcam_with_overlay(filepath, heatmap, overlay_path, predicted_label)
            overlay_path_return = overlay_path
        else:
            overlay_path_return = None

        # Lưu kết quả vào cơ sở dữ liệu
        conn = connect_to_sql_server()
        if conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO diagnosis_results (username, name, email, phone_number, filename, predicted_label, confidence, grad_cam_path, overlay_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (session.get('username'), patient_name, email, phone_number, filename, predicted_label, confidence, gradcam_path, overlay_path_return))

            conn.commit()
            cursor.close()
            conn.close()

        # Trả về JSON với kết quả
        return jsonify({
            'prediction': predicted_label,
            'confidence': f"{confidence * 100:.2f}%",
            'image_path': filepath,
            'grad_cam_path': gradcam_path,
            'overlay_path': overlay_path_return  # Chỉ trả về overlay nếu có
        })

    return jsonify({'error': 'Invalid file format'}), 400

# Route hiển thị hình ảnh từ backend/uploads
@app.route('/show_images')
def show_images():
    # Lấy tất cả các file trong thư mục uploads
    image_files = [f for f in os.listdir(UPLOAD_FOLDER) if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))]
    
    # Trả về template hiển thị danh sách hình ảnh
    return render_template('dataset_stoke.html', image_files=image_files)

@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    # Trả về hình ảnh từ thư mục uploads
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/results')
def show_results():
    conn = connect_to_sql_server()
    if conn:
        cursor = conn.cursor()
        # Bỏ cột `username` khỏi query SQL
        cursor.execute("""
            SELECT name, email, phone_number, filename, predicted_label, confidence, grad_cam_path, overlay_path, diagnosis_date 
            FROM diagnosis_results
        """)
        results = cursor.fetchall()  # Lấy tất cả kết quả từ truy vấn
        cursor.close()
        conn.close()

        results_list = []
        for result in results:
            results_list.append({
                'name': result[0],
                'email': result[1],
                'phone_number': result[2],
                'filename': result[3],
                'predicted_label': result[4],
                'confidence': result[5],
                'grad_cam_path': result[6],
                'overlay_path': result[7],
                'diagnosis_date': result[8]
            })

        print(results_list)  # In kết quả để kiểm tra

        return render_template('story_result.html', results=results_list)
    else:
        return "Không thể kết nối đến cơ sở dữ liệu"

# Route đăng xuất
@app.route('/logout')
def logout():
    # Xóa session của người dùng
    session.pop('user_id', None)
    session.pop('username', None)
    return redirect(url_for('login'))  # Chuyển hướng về trang login

@app.route('/statistics', methods=['GET', 'POST'])
def statistics():
    conn = connect_to_sql_server()
    if conn:
        cursor = conn.cursor()

        # Thống kê tổng số trường hợp
        cursor.execute("SELECT COUNT(*) FROM diagnosis_results")
        total_cases = cursor.fetchone()[0]

        # Lọc theo ngày (nếu có chọn lọc ngày)
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')

        # Xây dựng điều kiện lọc
        filters = []

        if start_date and end_date:
            filters.append(f"CONVERT(date, diagnosis_date) BETWEEN '{start_date}' AND '{end_date}'")
        
        # Lọc theo loại bệnh (predicted_label) nếu có
        selected_label = request.form.get('predicted_label')
        if selected_label:
            filters.append(f"predicted_label = '{selected_label}'")

        # Xây dựng phần điều kiện SQL, nếu không có filter thì không thêm "WHERE"
        filter_condition = ""
        if filters:
            filter_condition = "WHERE " + " AND ".join(filters)

        # Truy vấn thống kê theo ngày
        cursor.execute(f"""
            SELECT CONVERT(date, diagnosis_date) AS date, COUNT(*) 
            FROM diagnosis_results
            {filter_condition}
            GROUP BY CONVERT(date, diagnosis_date)
            ORDER BY date DESC
        """)
        daily_statistics = cursor.fetchall()

        # Thống kê theo loại (predicted_label)
        cursor.execute(f"""
            SELECT predicted_label, COUNT(*) 
            FROM diagnosis_results
            {filter_condition}
            GROUP BY predicted_label
            ORDER BY COUNT(*) DESC
        """)
        label_statistics = cursor.fetchall()

        # Lấy labels và counts cho biểu đồ tròn
        labels = [row[0] for row in label_statistics]
        counts = [row[1] for row in label_statistics]

        cursor.close()
        conn.close()

        # Trả về kết quả thống kê
        return render_template('story_search.html', 
                               total_cases=total_cases, 
                               daily_statistics=daily_statistics, 
                               label_statistics=label_statistics, 
                               labels=labels, 
                               counts=counts, 
                               start_date=start_date, 
                               end_date=end_date, 
                               selected_label=selected_label)
    else:
        return "Không thể kết nối đến cơ sở dữ liệu"
@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))  # Nếu chưa đăng nhập thì chuyển hướng đến trang login
    return render_template('index.html', username=session.get('username'))


if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
