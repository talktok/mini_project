from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import uuid
import random # AI 평가 더미 데이터를 위한 임시 import

app = Flask(__name__)

app.config['SECRET_KEY'] = 'your_super_secret_key_for_session_security'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 허용된 파일 확장자
ALLOWED_EXTENSIONS = {'ppt', 'pptx'}
ALLOWED_AUDIO_EXTENSIONS = {'mp3', 'wav', 'webm'}

# 최대 파일 크기 (16MB for PPT, 50MB for audio)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB for audio files
app.config['MAX_AUDIO_FILE_SIZE'] = 50 * 1024 * 1024  # 50MB

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def get_id(self):
        return str(self.id)

    def __repr__(self):
        return f"User('{self.username}', '{self.email}')"

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

def allowed_file(filename):
    """허용된 파일 확장자인지 확인"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_audio_file(filename):
    """허용된 오디오 파일 확장자인지 확인"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_AUDIO_EXTENSIONS

# --- 메인 페이지 라우트 ---
@app.route('/')
def index():
    return render_template('index.html', is_authenticated=current_user.is_authenticated)

@app.route('/upload', methods=['POST'])
def upload():
    if 'user_id' not in session:                       # ← 세션 검사
        return jsonify(success=False, message='로그인이 필요합니다.'), 401
    file = request.files.get('pptx')
    # …(유효성 검사 & 저장)…
    return jsonify(success=True, message='업로드 성공!')

# --- 회원가입 라우트 ---
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        existing_user_username = db.session.execute(db.select(User).filter_by(username=username)).scalar_one_or_none()
        existing_user_email = db.session.execute(db.select(User).filter_by(email=email)).scalar_one_or_none()

        if existing_user_username:
            flash('사용자명이 이미 존재합니다. 다른 사용자명을 선택해주세요.', 'danger')
            return redirect(url_for('register'))
        if existing_user_email:
            flash('이메일이 이미 사용 중입니다. 다른 이메일을 사용해주세요.', 'danger')
            return redirect(url_for('register'))

        new_user = User(username=username, email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        flash('회원가입이 완료되었습니다! 이제 로그인할 수 있습니다.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

# --- 로그인 라우트 ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        remember = True if request.form.get('remember_me') else False

        user = db.session.execute(db.select(User).filter_by(username=username)).scalar_one_or_none()

        if user and user.check_password(password):
            login_user(user, remember=remember)
            flash('로그인되었습니다!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page or url_for('dashboard'))
        else:
            flash('로그인 실패. 사용자명 또는 비밀번호를 확인해주세요.', 'danger')
    return render_template('login.html')

# --- 로그아웃 라우트 ---
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('로그아웃되었습니다.', 'info')
    return redirect(url_for('index'))

# --- 대시보드 라우트 (로그인 필요) ---
@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

# --- 스크립트 생성 흐름 시작 (사용자 정보 입력) ---
@app.route('/script_generator/user_info', methods=['GET', 'POST'])
def user_info_form():
    is_fresh_start = False

    if request.args.get('reset'):
        session.pop('user_profile', None)
        session.pop('ppt_filename_for_script', None)
        session.pop('current_slide_index', None)
        session.pop('all_selected_keywords', None)
        session.pop('script_options', None)
        session.pop('generated_script_text', None)
        session.pop('recorded_audio_filename', None)
        is_fresh_start = True

    if request.method == 'POST':
        presenter = request.form.get('presenter')
        audience = request.form.get('audience')
        purpose = request.form.get('purpose')
        tone = request.form.get('tone')
        time = request.form.get('time')

        if not all([presenter, audience, purpose, tone, time]):
            flash('모든 사용자 정보를 입력해주세요.', 'danger')
            return render_template('user_info_form.html', user_profile=request.form, is_fresh_start=is_fresh_start)

        session['user_profile'] = {
            'presenter': presenter,
            'audience': audience,
            'purpose': purpose,
            'tone': tone,
            'time': time
        }
        flash('사용자 정보가 저장되었습니다. 이제 발표 자료 (PPT)를 업로드해주세요.', 'success')
        return redirect(url_for('upload_ppt_for_script'))

    user_profile = session.get('user_profile', {})
    return render_template('user_info_form.html', user_profile=user_profile, is_fresh_start=is_fresh_start)

# --- 스크립트 생성을 위한 PPT 업로드 (향상된 버전) ---
@app.route('/script_generator/upload_ppt_for_script', methods=['GET', 'POST'])
def upload_ppt_for_script():
    if 'user_profile' not in session:
        flash('먼저 사용자 정보를 입력해주세요.', 'danger')
        return redirect(url_for('user_info_form'))

    if request.method == 'POST':
        # 파일 존재 여부 확인
        if 'file' not in request.files:
            flash('파일이 업로드되지 않았습니다.', 'danger')
            return redirect(url_for('upload_ppt_for_script'))

        file = request.files['file']
        
        # 파일 선택 여부 확인
        if file.filename == '':
            flash('파일을 선택해주세요.', 'danger')
            return redirect(url_for('upload_ppt_for_script'))

        # 파일 확장자 검증
        if not allowed_file(file.filename):
            flash('PPT 파일 (.ppt, .pptx)만 업로드할 수 있습니다.', 'danger')
            return redirect(url_for('upload_ppt_for_script'))

        # 파일 크기 검증 (클라이언트 측에서도 확인하지만 서버 측에서도 재확인)
        if file.content_length and file.content_length > 16 * 1024 * 1024:  # 16MB for PPT
            flash('PPT 파일 크기가 너무 큽니다. 최대 16MB까지 업로드 가능합니다.', 'danger')
            return redirect(url_for('upload_ppt_for_script'))

        try:
            # 안전한 파일명 생성
            original_filename = secure_filename(file.filename)
            file_extension = os.path.splitext(original_filename)[1]
            unique_filename = str(uuid.uuid4()) + file_extension
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            # 파일 저장
            file.save(filepath)
            
            # 파일 저장 확인
            if not os.path.exists(filepath):
                flash('파일 저장 중 오류가 발생했습니다. 다시 시도해주세요.', 'danger')
                return redirect(url_for('upload_ppt_for_script'))
            
            # 세션에 파일 정보 저장
            session['ppt_filename_for_script'] = unique_filename
            session['ppt_original_filename'] = original_filename
            
            flash(f'"{original_filename}" 파일이 성공적으로 업로드되었습니다. 이제 슬라이드별 키워드를 선택해주세요.', 'success')
            return redirect(url_for('keyword_select'))
            
        except Exception as e:
            flash('파일 업로드 중 오류가 발생했습니다. 다시 시도해주세요.', 'danger')
            return redirect(url_for('upload_ppt_for_script'))

    return render_template('script_generator.html')

# --- 파일 크기 초과 오류 처리 ---
@app.errorhandler(413)
def too_large(e):
    flash('파일 크기가 너무 큽니다. PPT 파일은 최대 16MB, 오디오 파일은 최대 50MB까지 업로드 가능합니다.', 'danger')
    return redirect(request.url)

# --- 스크립트 생성 (슬라이드별 키워드 선택) ---

SLIDE_KEYWORDS_DATA = [
    ['인공지능', 'AI 혁신', '미래 기술', '자동화', '데이터 분석'],
    ['머신러닝', '딥러닝', '알고리즘', '예측 모델', '학습 데이터'],
    ['클라우드 컴퓨팅', 'SaaS', 'PaaS', 'IaaS', '분산 시스템'],
    ['사물 인터넷', 'IoT 기기', '연결성', '스마트 홈', '스마트 시티'],
    ['블록체인', '분산원장', '암호화폐', '보안', '투명성'],
]
TOTAL_SLIDES = len(SLIDE_KEYWORDS_DATA)

@app.route('/script_generator/keywords', methods=['GET', 'POST'])
def keyword_select():
    if 'user_profile' not in session or 'ppt_filename_for_script' not in session:
        flash('먼저 사용자 정보와 PPT 파일을 업로드해주세요.', 'danger')
        return redirect(url_for('user_info_form', reset=True))

    if 'current_slide_index' not in session or request.args.get('reset'):
        session['current_slide_index'] = 0
        session['all_selected_keywords'] = []
        flash('슬라이드별 키워드를 선택해주세요.', 'info')

    current_slide_index = session.get('current_slide_index')
    
    if current_slide_index >= TOTAL_SLIDES:
        flash('모든 슬라이드의 키워드 선택이 완료되었습니다.', 'success')
        return redirect(url_for('script_option'))

    current_slide_keywords = SLIDE_KEYWORDS_DATA[current_slide_index]

    if request.method == 'POST':
        selected_keywords_for_current_slide = request.form.getlist('keywords')
        if not selected_keywords_for_current_slide:
            flash('현재 슬라이드에 대한 키워드를 하나 이상 선택해주세요.', 'danger')
            return render_template('keyword_select.html',
                                   current_slide_index=current_slide_index,
                                   total_slides=TOTAL_SLIDES,
                                   keywords=current_slide_keywords)
        
        all_selected_keywords = session.get('all_selected_keywords', [])
        all_selected_keywords.append(selected_keywords_for_current_slide)
        session['all_selected_keywords'] = all_selected_keywords
        
        session['current_slide_index'] += 1
        flash(f'슬라이드 {current_slide_index + 1}의 키워드가 저장되었습니다.', 'success')
        return redirect(url_for('keyword_select'))
    
    return render_template('keyword_select.html',
                           current_slide_index=current_slide_index,
                           total_slides=TOTAL_SLIDES,
                           keywords=current_slide_keywords)

# --- 스크립트 옵션 선택 (서론/본론/결론) ---
@app.route('/script_generator/options', methods=['GET', 'POST'])
def script_option():
    if 'user_profile' not in session or 'all_selected_keywords' not in session or len(session['all_selected_keywords']) < TOTAL_SLIDES:
        flash('스크립트 생성에 필요한 정보가 부족합니다. 처음부터 다시 시작해주세요.', 'danger')
        return redirect(url_for('user_info_form', reset=True))

    all_selected_keywords_by_slide = session.get('all_selected_keywords')
    selected_keywords_flat = [item for sublist in all_selected_keywords_by_slide for item in sublist]
    
    if request.method == 'POST':
        intro_option = request.form.get('intro_option')
        body_option = request.form.get('body_option')
        conclusion_option = request.form.get('conclusion_option')

        if not all([intro_option, body_option, conclusion_option]):
            flash('서론, 본론, 결론 옵션을 모두 선택해주세요.', 'danger')
            intro_options = ['흥미로운 질문으로 시작', '최신 트렌드 언급', '문제 제기']
            body_options = ['사례 연구 포함', '기술적 설명 강조', '미래 전망 제시']
            conclusion_options = ['핵심 요약 및 제언', '청중에게 질문 던지기', '긍정적 비전 제시']
            return render_template('script_option.html',
                                   selected_keywords=selected_keywords_flat,
                                   intro_options=intro_options,
                                   body_options=body_options,
                                   conclusion_options=conclusion_options)

        session['script_options'] = {
            'intro': intro_option,
            'body': body_option,
            'conclusion': conclusion_option
        }
        flash('스크립트 옵션이 저장되었습니다.', 'success')
        return redirect(url_for('script_result'))

    intro_options = ['흥미로운 질문으로 시작', '최신 트렌드 언급', '문제 제기']
    body_options = ['사례 연구 포함', '기술적 설명 강조', '미래 전망 제시']
    conclusion_options = ['핵심 요약 및 제언', '청중에게 질문 던지기', '긍정적 비전 제시']

    return render_template('script_option.html', 
                           selected_keywords=selected_keywords_flat,
                           intro_options=intro_options,
                           body_options=body_options,
                           conclusion_options=conclusion_options)

# --- 완성된 대본 보여주기 ---
@app.route('/script_generator/result')
def script_result():
    user_profile = session.get('user_profile')
    all_selected_keywords_by_slide = session.get('all_selected_keywords')
    script_options = session.get('script_options')

    if not user_profile or not all_selected_keywords_by_slide or not script_options:
        flash('스크립트 생성에 필요한 정보가 부족합니다. 처음부터 다시 시작해주세요.', 'danger')
        return redirect(url_for('user_info_form', reset=True))

    selected_keywords_flat = [item for sublist in all_selected_keywords_by_slide for item in sublist]

    # --- 대본 생성 로직 (PLACEHOLDER) ---
    original_filename = session.get('ppt_original_filename', '업로드된 파일')
    generated_script = f"""
    --- 대본 시작 ---

    **업로드된 파일:** {original_filename}
    
    **발표자 정보:**
    발표자: {user_profile.get('presenter', '미상')}
    발표 대상: {user_profile.get('audience', '미상')}
    발표 목적: {user_profile.get('purpose', '미상')}
    말투 선호: {user_profile.get('tone', '미상')}
    발표 시간: {user_profile.get('time', '미상')}

    **선택된 슬라이드별 키워드:**
    """
    for i, slide_keywords in enumerate(all_selected_keywords_by_slide):
        generated_script += f"\n슬라이드 {i+1}: {', '.join(slide_keywords)}"

    generated_script += f"""

    **서론:**
    선택된 서론 옵션: "{script_options.get('intro', '없음')}"
    안녕하세요! 저는 {user_profile.get('presenter', '발표자')}입니다.
    오늘 {user_profile.get('audience', '여러분')}께 {", ".join(selected_keywords_flat)}와 관련된 주제로 발표를 진행하겠습니다.
    이번 발표를 통해 {user_profile.get('purpose', '좋은 정보를 전달')}하고자 합니다.
    {script_options.get('intro', '시작하는 인사말')}.

    **본론:**
    선택된 본론 옵션: "{script_options.get('body', '없음')}"
    핵심 개념과 중요성을 {user_profile.get('tone', '부드럽고 자연스럽게')} 설명합니다.
    관련 사례를 들어 내용을 구체화합니다.
    (이 부분에 슬라이드별 키워드를 활용한 상세 내용이 들어갈 수 있습니다.)

    **결론:**
    선택된 결론 옵션: "{script_options.get('conclusion', '없음')}"
    오늘 발표를 {user_profile.get('time', '5분')} 동안 요약하고, 미래 방향성을 제시합니다.
    {script_options.get('conclusion', '마무리 인사말')}.

    --- 대본 끝 ---
    """
    
    session['generated_script_text'] = generated_script

    return render_template('script_result.html', 
                           generated_script=generated_script,
                           selected_keywords=selected_keywords_flat,
                           all_selected_keywords_by_slide=all_selected_keywords_by_slide,
                           script_options=script_options,
                           user_profile=user_profile)

# --- 대본 녹음 파일 업로드 페이지 ---
@app.route('/script_generator/record', methods=['GET', 'POST'])
def record_script():
    generated_script = session.get('generated_script_text')
    if not generated_script:
        flash('먼저 대본을 생성해주세요.', 'danger')
        return redirect(url_for('user_info_form', reset=True))

    if request.method == 'POST':
        # 파일 존재 여부 확인
        if 'audio_data' not in request.files:
            flash('오디오 파일이 업로드되지 않았습니다.', 'danger')
            return redirect(url_for('record_script'))

        audio_file = request.files['audio_data']
        
        # 파일 선택 여부 확인
        if audio_file.filename == '':
            flash('오디오 파일을 선택해주세요.', 'danger')
            return redirect(url_for('record_script'))

        # 파일 확장자 검증
        if not allowed_audio_file(audio_file.filename):
            flash('지원하지 않는 오디오 형식입니다. .mp3, .wav, .webm 파일을 업로드해주세요.', 'danger')
            return redirect(url_for('record_script'))

        # 파일 크기 검증 (50MB)
        if audio_file.content_length and audio_file.content_length > app.config['MAX_AUDIO_FILE_SIZE']:
            flash('오디오 파일 크기가 너무 큽니다. 최대 50MB까지 업로드 가능합니다.', 'danger')
            return redirect(url_for('record_script'))

        try:
            # 안전한 파일명 생성
            original_filename = secure_filename(audio_file.filename)
            file_extension = os.path.splitext(original_filename)[1]
            unique_filename = str(uuid.uuid4()) + file_extension
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            # 파일 저장
            audio_file.save(filepath)
            
            # 파일 저장 확인
            if not os.path.exists(filepath):
                flash('오디오 파일 저장 중 오류가 발생했습니다. 다시 시도해주세요.', 'danger')
                return redirect(url_for('record_script'))

            # 세션에 파일 정보 저장
            session['recorded_audio_filename'] = unique_filename
            session['recorded_audio_original_filename'] = original_filename
            
            flash('녹음 파일이 성공적으로 업로드되었습니다. 이제 AI 평가를 시작할 수 있습니다.', 'success')
            return redirect(url_for('evaluation_result'))
            
        except Exception as e:
            flash('오디오 파일 업로드 중 오류가 발생했습니다. 다시 시도해주세요.', 'danger')
            return redirect(url_for('record_script'))

    # GET 요청 시, 대본 텍스트를 템플릿으로 전달합니다.
    return render_template('record_script.html', generated_script=generated_script)

# --- AI 평가 결과 페이지 ---
@app.route('/script_generator/evaluation')
def evaluation_result():
    recorded_audio_filename = session.get('recorded_audio_filename')
    generated_script = session.get('generated_script_text')
    user_profile = session.get('user_profile')

    if not recorded_audio_filename or not generated_script:
        flash('평가에 필요한 정보(녹음 파일 또는 대본)가 부족합니다. 다시 시도해주세요.', 'danger')
        return redirect(url_for('user_info_form', reset=True))

    audio_filepath = os.path.join(app.config['UPLOAD_FOLDER'], recorded_audio_filename)

    # --- AI 평가 더미 로직 ---
    evaluation_scores = {
        'pronunciation': random.randint(70, 95),
        'speed': random.randint(60, 90),
        'emphasis': random.randint(50, 85),
        'confidence': random.randint(75, 98),
        'fluency': random.randint(65, 90)
    }

    overall_score = sum(evaluation_scores.values()) / len(evaluation_scores)

    feedback_messages = []
    if evaluation_scores['pronunciation'] < 80:
        feedback_messages.append("일부 단어의 발음이 불분명할 수 있습니다. 반복 연습이 필요합니다.")
    if evaluation_scores['speed'] < 70:
        feedback_messages.append("발표 속도가 다소 빠르거나 느릴 수 있습니다. 청중의 이해를 위해 속도 조절을 고려해보세요.")
    elif evaluation_scores['speed'] > 85:
        feedback_messages.append("발표 속도가 적절합니다. 다만, 중요한 부분에서는 약간의 속도 변화를 주면 더욱 좋습니다.")
    if evaluation_scores['emphasis'] < 70:
        feedback_messages.append("핵심 메시지 강조가 부족할 수 있습니다. 중요한 단어에 힘을 주어 말하는 연습을 해보세요.")
    if evaluation_scores['confidence'] < 80:
        feedback_messages.append("억양에서 자신감이 부족하게 느껴질 수 있습니다. 확신을 가지고 말하는 연습을 해보세요.")
    if evaluation_scores['fluency'] < 75:
        feedback_messages.append("스크립트와 일치율이 낮거나, 발화가 부자연스러운 부분이 있습니다. 대본을 충분히 숙지해주세요.")
    
    if not feedback_messages:
        feedback_messages.append("전반적으로 훌륭한 발표였습니다! 계속해서 연습하시면 더욱 완벽해질 것입니다.")

    return render_template('evaluation_result.html',
                           overall_score=round(overall_score, 1),
                           evaluation_scores=evaluation_scores,
                           feedback_messages=feedback_messages,
                           generated_script=generated_script,
                           user_profile=user_profile)


# --- 데이터베이스 생성 (최초 1회만 실행) ---
if __name__ == '__main__':
    with app.app_context():
       db.create_all()
    app.run(debug=True)