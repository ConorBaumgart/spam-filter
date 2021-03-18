<template>
  <v-container>
    <v-row class="text-center">
      <v-col cols="12">
        <v-img
          :src="require('../assets/spammail.jpg')"
          class="my-3"
          contain
          height="200"
        />
      </v-col>

      <v-col class="mb-4">
        <h1 class="display-2 font-weight-bold mb-3">
          Is your message spam?
        </h1>

        <p class="subheading font-weight-regular">
          Use machine learning to analyze your message body
          <br>to determine if your message is spam or not.
        </p>

        <h2 v-if="showSubmit"><b>Paste your message below to find out!</b></h2>

        <Loading v-if="showLoading" />
        <v-container fluid id="message-input">
          <v-textarea
            name="input-7-1"
            solo
            label="This man has a billion dollar idea that he's willing to sell for only $20. Don't miss out!"
            auto-grow
            v-if="showSubmit"
            v-model="spamOrHam"
          ></v-textarea>
        </v-container>

        <v-container>
          <SpamAnswer v-bind:spamData="spamData" v-if="spamData !== ''" class="mb-4"/>
        </v-container>

        <v-btn v-if="showSubmit" v-on:click="submitMessage"
          elevation="2" color="#1A237E" dark
        ><b>Submit</b></v-btn>

        <v-btn v-if="!showSubmit" v-on:click="newMessage"
          elevation="2" color="#1A237E" dark
        ><b>Try Again</b></v-btn>

      </v-col>
    </v-row>
  </v-container>
</template>

<script>
  const axios = require('axios')
  import Loading from "./Loading"
  import SpamAnswer from "./SpamAnswer"

  export default {
    name: 'MessageInput',
    methods: {
      submitMessage() {
        (async () => {
          try {
            this.showSubmit = false
            this.showLoading = true

            if (this.spamOrHam === "") {
              window.alert("Do not submit an empty message")
              this.showSubmit = true
              this.showLoading = false
            } else {
              const response = await axios.post('http://localhost:5000/spamorham', {
                message: this.spamOrHam
              })
              this.spamData = [response.data]
            }
            this.showLoading = false
          } catch(error) {
            console.log(error)
            this.showSubmit = true
            this.showLoading = false
          }
          this.spamOrHam = ""
        }
      )()},
      newMessage() {
        this.showSubmit = true
        this.spamData = ""
      }
    },
    components: {
      Loading,
      SpamAnswer,
    },

    data: () => ({
      showSubmit: true,
      spamOrHam: "",
      showLoading: false,
      spamData: "",
    }),
  }
</script>

<style>
#message-input {
  max-width: 1000px;
}

</style>