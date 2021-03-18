<template>
  <v-app>
    <v-app-bar
      app
      color="#1A237E"
      dark
      prominant
    >
      <v-spacer />
      <div class="d-flex align-center">
        <v-img
          alt="Ham"
          class="shrink mr-2"
          contain
          src="@/assets/ham.jpg"
          transition="scale-transition"
          width="60"
        />

        <h1>Spam or Ham?</h1>

        <v-img
          alt="Ham"
          class="shrink ml-2"
          contain
          src="@/assets/spam.jpg"
          transition="scale-transition"
          width="60"
        />
      </div>
      <v-spacer />
    </v-app-bar>

    <v-main>
      <MessageInput/>
    </v-main>
  </v-app>
</template>

<script>
import MessageInput from './components/MessageInput'

export default {
  name: 'App',

  components: {
    MessageInput,
  }
}
</script>
