# Materials

量子情報を学ぶための教材

## Quantum Algorithm

* [Quantum Native Dojo](https://github.com/qulacs/quantum-native-dojo)
  * QunaSysが提供する初学者のための量子コンピュータ初学者のための自習教材

## Mathematics

* [Numerical Linear Algebra for Coders](https://github.com/fastai/numerical-linear-algebra/blob/master/README.md) 
  * 用例を通じて学ぶ線形代数。Numpy, scikit-learn, PyTorchを利用した実装を通じてその背後の仕組みを学ぶというスタイル。
* [Introduction to Applied Linear Algebra – Vectors, Matrices, and Least Squares](http://web.stanford.edu/~boyd/vmls/)
  * ケンブリッジで使用されている線形代数の教科書。実際にどんなところで利用されているのか、という解説までついていてわかりやすい。

## Optimization Method

* [An overview of gradient descent optimization algorithms](https://arxiv.org/abs/1609.04747)
  * SGDを端緒とした、最適化アルゴリズムについての解説
* [A birds-eye view of optimization algorithms](http://fa.bianp.net/teaching/2018/eecs227at/)
  * 最適化の手法について、図解付きの解説。


## Engineering

実システムへの組み込みや組み込んだ後の運用も大きな課題となります。また、そもそもどう仕様を決めるのか、と言った点も大きな問題です。それについて学ぶための資料について記載します。


### Project Planning

* [現代的システム開発概論](https://speakerdeck.com/rtechkouhou/xian-dai-de-sisutemukai-fa-gai-lun)
  * プロジェクトの計画、修正について体系的に書かれた資料。システム開発を行う前にまず参照しておくべき資料。

### Architecture

* [Web Application Tutorial](https://docs.google.com/presentation/d/1whFnASJKNTblT6o2vF84Cd0j8vhICouXcJAnBdGmMCw/edit?usp=sharing)
  * 基本的なMVCのアーキテクチャとそれを利用する際の注意点について解説しています。

### Development

#### Coding

* [python_exercises](https://github.com/icoxfog417/python_exercises)
  * Pythonのトレーニング用リポジトリです
* [良いコードとは](https://www.slideshare.net/nbykmatsui/ss-55961899)
  * 動けばいいというコードでは、自分の実験の生産性が落ちる可能性があるだけでなく、他の人が再現を行うのも難しくなります。良いコードを書くよう心がけましょう。


#### Git

アプリケーションの開発だけでなく、機械学習モデルのソースコードの管理にもバージョン管理ツールは欠かせません。  
ここでは、数あるバージョン管理ツールのうちGitに絞り資料を紹介します。

* [使い始める Git](https://qiita.com/icoxfog417/items/617094c6f9018149f41f)
  * 特定のファイルをバージョン管理対象外にする`.gitignore`は必ず確認しましょう。よく、`.pyc`ファイルや`.ipynb_checkpoints`がリポジトリに入ってしまっている例を見ます。[こちら](https://github.com/github/gitignore)で言語や開発環境に応じたファイルを確認できます。
* [Try Git](https://try.github.io/levels/1/challenges/1)
  * GitHubオフィシャルのGitチュートリアルです

#### Docker

量子情報エンジニアにとってDockerはもはや欠かせないツールになっているので、理解しておくとよいです。

* [コンテナ未経験新人が学ぶコンテナ技術入門](https://www.slideshare.net/KoheiTokunaga/ss-122754942)
  * VMからDocker、Kubernetesに到るまでの過程と周辺技術要素がとてもよくまとめられた資料。この資料だけで、仕組みの理解は済んでしまうと思う。
* [プログラマのためのDocker教科書 インフラの基礎知識＆コードによる環境構築の自動化](https://www.amazon.co.jp/dp/B017UGA7NG)


#### Visualization

* [DataVisualization](https://github.com/neerjad/DataVisualization)
  * 実際のデータを利用した、データ可視化チュートリアル。各種ライブラリ(Seaborn/Bokeh/Plotly/Igraph)ごとに用意されていて使い方を比較できる。
* [Visual Vocabulary](https://gramener.github.io/visual-vocabulary-vega/)
  * データの可視化を行う技法を、可視化したいメトリクス(差なのか共起なのかetc)に応じて分類、紹介してくれているサイト。
* [UW Interactive Data Lab](https://idl.cs.washington.edu/)
  * データの可視化による情報伝達について、実例などをまとめているサイト。

## Others

* [From zero to research — An introduction to Meta-learning](https://medium.com/huggingface/from-zero-to-research-an-introduction-to-meta-learning-8e16e677f78a)
  * メタラーニングについて、初歩的なところから解説をしている記事。PyTorchでの実装例も付属している。アニメーションを使った図解が豊富でとても分かりやすい。


### How to Write

* [松尾ぐみの論文の書き方](http://ymatsuo.com/japanese/ronbun_jpn.html)
  * 論文を書く前に、まずはこちらに目を通しておいた方が良い。
* [Stanford大学流科学技術論文の書き方](http://hontolab.org/tips-for-research-activity/tips-for-writing-technical-papers/)
* [Good Citizen of CVPR](https://www.cc.gatech.edu/~parikh/citizenofcvpr/)
  * CVPRで行われた、良き研究者になるためのガイド的なワークショップの資料。論文の書き方からTodoの管理といった細かいところまで、多くの資料がある。

