# Tools

機械学習モデルの実装を行う際に役立つツールについてまとめています。  
なお、深層学習系のフレームワーク(TensorFlow/Chainerなど・・・)は既知だと思うので、その当たりは除外しています。

# Vision

* [OpenFace](https://cmusatyalab.github.io/openface/)
  * 画像から顔を認識し、128次元の特徴量を算出するツール。この特徴を利用し、独自の分類機などを作ることができる。
  * 顔の検知には、OpenCV/dlibを使用
  * Apache 2.0 License
* [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
  * 体のキーポイント(関節など)を検知するためのライブラリ
  * OpenCV/Caffeベース
  * ライセンスは独自のもので、商用利用については応相談
* [self-driving-car](https://github.com/udacity/self-driving-car/)
  * Udacityで自動運転車の講座が開講されるに伴い公開された、オープンソースの自動運転車の実装
  * 他の公開されている実装としては、[comma.ai](https://github.com/commaai/research)、Baiduが公開した[Appolo](http://apollo.auto/index.html)などがある。
* [europilot](https://github.com/marshq/europilot)
  * オープンソースの自動運転シミュレーター。Euro Truck Simulatorというトラックシミュレーションゲームと接続して動作する。
* [YOLO](https://pjreddie.com/darknet/yolo/)
  * You only look once (YOLO)と名付けられた、リアルタイムでの物体認識を行うソフトウェア。
  * C言語性のDNNのフレームワーク[darknet](https://pjreddie.com/darknet/)で構築されている
* [OpenZoonz](https://opentoonz.github.io/)
  * 2Dアニメーションの制作ソフトウェア(ジブリでの使用実績もあり)だが、ラスタライズ画像をベクター画像に変換できるという機能がある。
  * sketch-RNNなど、ベクターデータが必要な場合に役に立つ。

# NLP

* [NLTK](http://www.nltk.org/)
  * 形態素解析や品詞タグ認識など、自然言語処理に関する基礎的な処理を行うライブラリ。微妙に日本語にも対応している。
* [spaCy](https://spacy.io/)
  * NLTKど同様だが、固有表現抽出などの機能も搭載している
  * 最大のポイントとしてはエンタープライズでの利用を想定しており、マルチスレッドを用い高速に動作するよう設計されている
* [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/)
  * Stanfordが提供する自然言語処理のライブラリ。基礎的なライブラリでできることは網羅されており、照応表現(彼=○○さん、とか)の解釈や、構文解析結果を利用した処理なども可能(名詞を対象にｘｘするなど)
  * 多様なプログラミング言語へのインタフェースを持つことも特徴で、[ClojureやErlang/Elixirなどまでサポートされている](https://stanfordnlp.github.io/CoreNLP/other-languages.html)
  * なお、CoreNLPも含めたStanfordが開発しているツールは[Stanford NLP Software](https://nlp.stanford.edu/software/)にまとまっている。自然言語系で何かないかなと思ったら、まず目を通してみると良い。
* [ParlAI](https://github.com/facebookresearch/ParlAI)
  * Facebookが発表した、対話モデルを開発するためのフレームワーク。
  * SQuADやbAbI、Ubuntu Dialogなど有名な質問応答/対話系のデータセットが簡単に扱えるようになっている
  * モデル評価のためにAmazon Mechanical Turkとの連携も兼ね備えるなど、対話研究を行うならとても便利なフレームワークになっている。
* [NeuroNER](http://neuroner.com/)
  * TensorFlowベースの固有表現抽出ツール。学習済みモデルが利用できるほか、追加データによる学習も可能。
* [OpenNMT](http://opennmt.net/)
  * オープンソースの機械翻訳の実装。Torch/PyTorchで実装されている。
  * 幾つかの言語間では、学習済みモデルも提供されている。
* [SentEval](https://github.com/facebookresearch/SentEval)
  * 文のベクトル表現の品質を計測するためのプラットフォーム。SentimentやQuestion-typeの推定など、様々なタスクで評価ができる。
* [FuzzyWuzzy](https://github.com/seatgeek/fuzzywuzzy)
  * 文の類似度スコアを計算してくれるツール(レーベンシュタイン距離をベースにしているよう)。表記ゆれなどを解消する際に役に立つ。
  * なぜかRustの実装もある。
* [chazutsu](https://github.com/chakki-works/chazutsu)
  * 自然言語関係のデータセットを簡単にダウンロードし、使えるようにできるツール
  * データをダウンロード、展開して、整形して・・・という作業は以外と時間がかかるもので、その個所をスキップできる
* [chakin](https://github.com/chakki-works/chakin)
  * 単語の分散表現を簡単にダウンロードできるツール。こちらも探してくるのが面倒なので、これを利用すれば簡単に利用できる。


**Additional**

* [日本語自動品詞分解ツール](http://tool.konisimple.net/text/hinshi_keitaiso)
  * オンラインでさっと品詞分解ができるツール。手軽なのでよく使う

# Audio

* [Magenta](https://github.com/tensorflow/magenta)
  * Googleが開発を進める、機械学習をアートに適用するためのプロジェクト。音楽生成や、スタイルトランスファーなどのコードが組み込まれている
  * 特に音声系では、MIDIファイルを扱うためのAPIなども提供されているので、単純にデモを動かすだけでなく開発のベースとしても利用ができる。
* [librosa](https://github.com/librosa/librosa)
  * 音声データを読み込み、特徴量(MFCCなど)を算出するなどを行ってくれるライブラリ。tempoの推定なども行える。
* [sprocket](https://github.com/k2kobayashi/sprocket)
  * 統計的音声変換を行うライブラリ。Python3ベース。
  * GMMを用いた基礎的な手法が実装されており、Voice Conversion Challenge2018でベースラインシステムとして採用されている。

# Reinforcement Learning

* [OpenAI Gym](https://gym.openai.com/)
  * 強化学習モデルのトレーニング環境を提供するフレームワーク
  * OpenAI Gym自体にも多くの環境(Atariのゲーム)などが入っているが、Open AI GymのAPIを備えたサードパーティー製の環境も追加でインストールできる([gym_pull](https://github.com/openai/gym/wiki/Environments#gym_pull)参照)。
* [OpenAI Universe](https://universe.openai.com/)
  * VNCを利用し、まさに人間が操作するようにPCを操作させられる強化学習のプラットフォーム。これによって、画面操作をエージェントに学習させることができる。
* [μniverse](https://github.com/unixpickle/muniverse)
  * OpenAI Universeの重たい所(VNCでリモート操作)、難しい所(画面がFlashで読み取りむずい)を改善したmuniverseが開発中とのこと。こちらはHTML5のゲームに限定することで、上記の問題点を回避している。
* [Microsoft/AirSim](https://github.com/microsoft/airsim)
  * Microsoftが公開したシミュレーター。自動運転や、ドローンといった実世界で動かすようなものの学習を行うことができる。
* [SerpentAI](https://github.com/SerpentAI/SerpentAI)
  * 手元のゲームを強化学習のための環境に使えるというライブラリ。Universeへの不満から生まれたとのこと。
  * ネイティブで動作し、ゲームに接続するAPIを作る機能が内包されているためゲーム側の対応を待つ必要がない。チュートリアルではSteamのゲームを使う手順を公開している。
* [malmo](https://github.com/Microsoft/malmo)
  * 協調するマルチエージェントの学習に特化した学習プラットフォーム。
  * Microsoftが公開しており、Minecraftの上で動く
  * Python、Lua、C#、C++、Javaなど多様なインタフェースを持つ
* [TorchCraft](https://github.com/TorchCraft/TorchCraft)
  * StarCraftを学習するためのフレームワーク
  * なぜかalibabaからPyTorchと連携できるようにするためのツールが公開されている([torchcraft-py](https://github.com/alibaba/torchcraft-py))
* [Learning to run](https://github.com/stanfordnmbl/osim-rl)
  * 観測情報から足の筋肉を操作するモデルを構築し、歩行を学習させるための環境。OpenAI Gym + Kerasで構築されている。
* [mujoco-py](https://github.com/openai/mujoco-py)
  * 物理エンジンであるMujocoをPythonから操作できるライブラリ  
* [pybullet](https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet)
  * Pythonから使える物理シミュレーター。OpenAI Gym/TensorFlowにも対応していて、これらを利用した歩行トレーニングを行う[チュートリアルも提供されている](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.2ye70wns7io3)。
* [Unity ML - Agents](https://github.com/Unity-Technologies/ml-agents)
  * Unityが公式に公開した、Unityで強化学習を行うための環境・エージェントを作成できるSDK
* [HoME-Platform](https://github.com/HoME-Platform/home-platform)
  * 家の中でエージェントを学習させられる学習環境。
  * 環境内のオブジェクトにはセグメンテーションや「棚の上の花瓶」といった言語情報がアノテートされており、音響環境も整えられている。もちろん物理エンジン搭載で、オブジェクトの移動などもシミュレートできる。
  * OpenAI Gymに対応

# Others

## Annotation Tool

* [visdial-amt-chat](https://github.com/batra-mlp-lab/visdial-amt-chat)
  * 画像に関する質問＋回答のデータを収集するためのアプリケーション。[実際のアプリケーション開発](https://visualdialog.org/)に使用したものを公開してくれている

## DataAPI

* [pandas](http://pandas.pydata.org/)
  * 表形式データを扱いやすくるためのツール。描画機能も付属しているため、簡単なグラフならpandasのみでかなりいける([こちら](http://pbpython.com/effective-matplotlib.html)参照)。
* [feather](https://github.com/wesm/feather)
  * 様々なプラットフォーム(Python/pandas、Rなど)で共通して扱え、なおかつ高速性を担保するデータフレームのファイルフォーマット、およびそれを扱うツール
  * いったんfeather形式のファイルフォーマットで保存し、それを読み込む形になる。この操作は特にメモリに乗らないような大規模なデータセットで力を発揮し、デモでは600MBのファイルを1秒たらずで読み込めている([Feather: A Fast On-Disk Format for Data Frames for R and Python, powered by Apache Arrow](https://blog.rstudio.org/2016/03/29/feather/))。

## Visualization

* [bokeh](http://bokeh.pydata.org/en/latest/)
  * Pythonのグラフ描画ツールで、描画結果からHTMLコンテンツの作成が可能。このため、Jupyter上で使うほかWebページに埋め込んだりすることもできる
* [seaborn](https://seaborn.pydata.org/)
  * Matplotlib上で稼働し、より使いやすく、きれいに描画できる
* [HyperTools](http://hypertools.readthedocs.io/en/latest/)
  * 高次元のデータの可視化に特化したツール
* [Picasso](https://medium.com/merantix/picasso-a-free-open-source-visualizer-for-cnns-d8ed3a35cfc5)
  * CNNの層を可視化するためのツール。Keras/TensorFlowのチェックポイントファイルをベースに動く。
* [scattertext](https://github.com/JasonKessler/scattertext)
  * コーパス内の単語を使って散布図を作れるビジュアライゼーションツール。どの分類にどの単語が効いているかなどを、視覚的にみることができる

## Official Implementation

ベースラインとして利用できるような、オフィシャルの実装を紹介します。

* [OpenAI Baselines](https://github.com/openai/baselines)
* [Memory-Augmented Neural Networks](https://github.com/facebook/MemNN)
* [Neural-Dialogue-Generation](https://github.com/jiweil/Neural-Dialogue-Generation)
* [PixelCNN++](https://github.com/openai/pixel-cnn)
* [benchmark_results](https://github.com/foolwood/benchmark_results)
  * Visual Trackingのベンチマーク集
* [chainerrl](https://github.com/chainer/chainerrl)
  * Chainerで実装された強化学習のアルゴリズム集
* [keras-rl](https://github.com/matthiasplappert/keras-rl)
  * Kerasで実装された強化学習のアルゴリズム集
