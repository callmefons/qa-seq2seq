var app = new Vue({
  el: '#app',
  data () {
    return {
      api: 'http://35.236.170.45/',
      faq: '',
      faqs: [],
      question: '',
      answer: '',
      gender: '女性',
      genders: [
        { text: '女性', value: '女性' },
        { text: '男性', value: '男性' }
      ],
      prefecture: '鹿児島県',
      prefectures: [
        '鹿児島県', '愛知県','長野県','三重県','福岡県','岡山県','東京都','静岡県','奈良県','大阪府','福島県','北海道','宮城県','千葉県','神奈川県','徳島県','京都府','埼玉県','茨城県','広島県','富山県','香川県','秋田県','岩手県','群馬県','兵庫県','岐阜県','滋賀県','沖縄県','宮崎県','愛媛県','福井県','石川県','佐賀県','和歌山県','新潟県','大分県','山口県','山梨県','島根県','鳥取県','高知県','山形県','栃木県','長崎県','青森県','熊本県'
      ],
      eval: {},
    }
  },

  methods: {
    
    ask: function () {
      axios
      .post(`${this.api}ask`,{
         question: this.question,
         gender: this.gender
      })
      .then(response => {
        this.answer = response.data.result
      })
      .catch(e => {
        console.log(e)
      })
    },

    clear: function (){
      this.question = ''
      this.answer = ''
    },

    select_faq: function(){
      this.question = this.faq
    },

    get_faqs(){
      axios
      .post(`${this.api}get_faqs`,{
         gender: this.gender,
         prefecture: this.prefecture
      })
      .then(response => {
        this.faq = response.data.result[0]
        this.faqs = response.data.result
      })
      .catch(e => {
        console.log(e)
      })
    },

    evaluate_randomly: function(){
      axios
      .get(`${this.api}evaluate_randomly`)
      .then(response => {
        var result= JSON.parse(response.data);
        this.eval = result
      })
      .catch(e => {
        console.log(e)
      })
    }


  },

  created: function(){
        this.get_faqs()
  }

})