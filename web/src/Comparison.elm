module Comparison exposing (..)

import Array
import Browser
import Browser.Navigation
import Dict
import File
import Gradio
import Html
import Html.Attributes
import Html.Events
import Json.Decode as D
import Json.Encode as E
import Requests
import Url


main =
    Browser.application
        { init = init
        , view = view
        , update = update
        , subscriptions = \model -> Sub.none
        , onUrlRequest = onUrlRequest
        , onUrlChange = onUrlChange
        }



-- MESSAGE


type Msg
    = NoOp
    | GotImage Requests.Id (Result Gradio.Error Example)
    | GotSaeActivations Requests.Id (Result Gradio.Error (Dict.Dict String (List SaeActivation)))
    | SelectExample String
    | FocusLatent String Int
    | BlurLatent String Int



-- MODEL


type alias Model =
    { -- Browser
      key : Browser.Navigation.Key

    -- State
    , inputExample : Requests.Requested Example

    -- Mapping from ViT name (string) to a list of activations. Changes as the input image changes.
    , saeActivations : Requests.Requested (Dict.Dict String (List SaeActivation))
    , focusedLatent : Maybe ( String, Int )

    -- APIs
    , gradio : Gradio.Config
    , imageRequestId : Requests.Id
    , saeActivationsRequestId : Requests.Id
    }


type alias SaeActivation =
    { model : String
    , latent : Int
    , activations : List Float -- Has nPatches floats

    -- TODO: add examples
    , examples : List HighlightedExample
    }


type alias Example =
    { url : Gradio.Base64Image
    , label : String
    }


type alias HighlightedExample =
    { original : Gradio.Base64Image
    , highlighted : Gradio.Base64Image
    , label : String
    , exampleId : String
    }


init : () -> Url.Url -> Browser.Navigation.Key -> ( Model, Cmd Msg )
init _ url key =
    let
        model =
            { -- Browser
              key = key
            , inputExample = Requests.Initial
            , saeActivations = Requests.Initial
            , focusedLatent = Nothing

            -- APIs
            , gradio =
                { host = "http://127.0.0.1:7860" }
            , imageRequestId = Requests.init
            , saeActivationsRequestId = Requests.init
            }
    in
    ( model, getImage model.gradio model.imageRequestId "inat21__train_mini__93571" )



-- UPDATE


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case msg of
        NoOp ->
            ( model, Cmd.none )

        GotImage id result ->
            if Requests.isStale id model.imageRequestId then
                ( model, Cmd.none )

            else
                case result of
                    Ok example ->
                        let
                            saeActivationsRequestId =
                                Requests.next model.saeActivationsRequestId
                        in
                        ( { model
                            | inputExample = Requests.Loaded example
                            , saeActivationsRequestId = saeActivationsRequestId
                            , saeActivations = Requests.Loading
                          }
                        , getSaeActivations model.gradio
                            saeActivationsRequestId
                            example.url
                        )

                    Err err ->
                        ( { model
                            | inputExample = Requests.Failed (explainGradioError err)
                          }
                        , Cmd.none
                        )

        GotSaeActivations id result ->
            if Requests.isStale id model.saeActivationsRequestId then
                ( model, Cmd.none )

            else
                case result of
                    Ok saeActivations ->
                        ( { model
                            | saeActivations = Requests.Loaded saeActivations
                          }
                        , Cmd.none
                        )

                    Err err ->
                        ( { model
                            | saeActivations = Requests.Failed (explainGradioError err)
                          }
                        , Cmd.none
                        )

        FocusLatent name latent ->
            ( { model | focusedLatent = Just ( name, latent ) }, Cmd.none )

        BlurLatent name latent ->
            case model.focusedLatent of
                Nothing ->
                    ( model, Cmd.none )

                Just ( n, l ) ->
                    if n == name && l == latent then
                        ( { model | focusedLatent = Nothing }, Cmd.none )

                    else
                        ( model, Cmd.none )

        SelectExample exampleId ->
            let
                imageRequestId =
                    Requests.next model.imageRequestId
            in
            ( { model
                | inputExample = Requests.Loading
                , imageRequestId = imageRequestId
                , saeActivations = Requests.Initial
                , focusedLatent = Nothing
              }
            , getImage model.gradio
                imageRequestId
                exampleId
            )


onUrlRequest : Browser.UrlRequest -> Msg
onUrlRequest request =
    NoOp


onUrlChange : Url.Url -> Msg
onUrlChange url =
    NoOp



-- API


explainGradioError : Gradio.Error -> String
explainGradioError err =
    case err of
        Gradio.NetworkError msg ->
            "Network error: " ++ msg

        Gradio.JsonError msg ->
            "Error decoding JSON: " ++ msg

        Gradio.ParsingError msg ->
            "Error parsing API response: " ++ msg

        Gradio.ApiError msg ->
            "Error in the API: " ++ msg


updateIds : String -> Dict.Dict String Requests.Id -> ( Requests.Id, Dict.Dict String Requests.Id )
updateIds path ids =
    let
        id =
            Dict.get path ids |> nextId
    in
    ( id, Dict.insert path id ids )


nextId : Maybe Requests.Id -> Requests.Id
nextId maybeId =
    maybeId
        |> Maybe.withDefault Requests.init
        |> Requests.next


getImage : Gradio.Config -> Requests.Id -> String -> Cmd Msg
getImage cfg id exampleId =
    Gradio.get cfg
        "get-image"
        [ E.string exampleId ]
        (D.map2 Example
            (D.index 0 Gradio.base64ImageDecoder)
            (D.index 1 D.string)
        )
        (GotImage id)


getSaeActivations : Gradio.Config -> Requests.Id -> Gradio.Base64Image -> Cmd Msg
getSaeActivations cfg id image =
    let
        latents =
            E.object
                [ ( "bioclip/inat21"
                  , E.list E.int
                        [ 449, 451, 518, 6448, 18380, 10185, 20085, 5435, 16081 ]
                  )
                , ( "clip/inat21"
                  , E.list E.int
                        [ 4661, 565, 5317 ]
                  )
                ]
    in
    Gradio.get cfg
        "get-sae-activations"
        [ Gradio.encodeImg image, latents ]
        (Gradio.decodeOne
            (D.dict
                (D.list
                    (D.map4
                        SaeActivation
                        (D.field "model_name" D.string)
                        (D.field "latent" D.int)
                        (D.field "activations" (D.list D.float))
                        (D.field "examples" (D.list highlightedExampleDecoder))
                    )
                )
            )
        )
        (GotSaeActivations id)


highlightedExampleDecoder : D.Decoder HighlightedExample
highlightedExampleDecoder =
    D.map4
        HighlightedExample
        (D.field "orig_url" Gradio.base64ImageDecoder)
        (D.field "highlighted_url" Gradio.base64ImageDecoder)
        (D.field "label" D.string)
        (D.field "example_id" D.string)



-- VIEW


view : Model -> Browser.Document Msg
view model =
    { title = "Image Classification"
    , body =
        [ Html.div
            [ Html.Attributes.class "flex flex-row" ]
            [ -- Image picker, left column
              Html.div
                [ Html.Attributes.class "flex flex-col" ]
                [ viewInputExample model.focusedLatent model.saeActivations model.inputExample
                , Html.p [] [ Html.text "TODO: other example images here" ]
                ]

            -- SAE stuff, right column
            , Html.div
                [ Html.Attributes.class "flex flex-col flex-1" ]
                [ viewSaeActivations model.focusedLatent model.saeActivations ]
            ]
        ]
    }


viewInputExample : Maybe ( String, Int ) -> Requests.Requested (Dict.Dict String (List SaeActivation)) -> Requests.Requested Example -> Html.Html Msg
viewInputExample focusedLatent requestedSaeActivations requestedExample =
    case ( focusedLatent, requestedSaeActivations, requestedExample ) of
        ( _, _, Requests.Initial ) ->
            Html.p
                []
                [ Html.text "Loading initial example..." ]

        ( _, _, Requests.Loading ) ->
            Html.p
                []
                [ Html.text "Loading example..." ]

        ( _, _, Requests.Failed err ) ->
            viewErr err

        ( Nothing, _, Requests.Loaded example ) ->
            viewGriddedImage [] example

        ( _, Requests.Initial, Requests.Loaded example ) ->
            viewGriddedImage [] example

        ( _, Requests.Loading, Requests.Loaded example ) ->
            viewGriddedImage [] example

        ( _, Requests.Failed _, Requests.Loaded example ) ->
            viewGriddedImage [] example

        ( Just ( model, latent ), Requests.Loaded activations, Requests.Loaded example ) ->
            let
                values =
                    Dict.get model activations
                        |> Maybe.map (List.filter (\act -> act.latent == latent))
                        |> Maybe.andThen List.head
                        |> Maybe.map .activations
                        |> Maybe.withDefault []
            in
            viewGriddedImage values example


viewGriddedImage : List Float -> Example -> Html.Html Msg
viewGriddedImage values { url, label } =
    Html.div
        []
        [ Html.div
            [ Html.Attributes.class "relative inline-block" ]
            [ Html.div
                [ Html.Attributes.class "absolute grid"
                , Html.Attributes.class "grid-rows-[repeat(14,_16px)] grid-cols-[repeat(14,_16px)]"
                , Html.Attributes.class "md:grid-rows-[repeat(14,_24px)] md:grid-cols-[repeat(14,_24px)]"
                , Html.Attributes.class "lg:grid-rows-[repeat(14,_32px)] lg:grid-cols-[repeat(14,_32px)]"
                , Html.Attributes.class "xl:grid-rows-[repeat(14,_40px)] xl:grid-cols-[repeat(14,_40px)]"
                ]
                (List.map viewGridCell values)
            , Html.img
                [ Html.Attributes.class "block w-[224px] h-[224px]"
                , Html.Attributes.class "md:w-[336px] md:h-[336px]"
                , Html.Attributes.class "lg:w-[448px] lg:h-[448px]"
                , Html.Attributes.class "xl:w-[560px] xl:h-[560px]"
                , Html.Attributes.src (Gradio.base64ImageToString url)
                ]
                []
            ]
        , Html.p
            []
            [ Html.text label ]
        ]


viewGridCell : Float -> Html.Html Msg
viewGridCell value =
    let
        opacity =
            Array.fromList
                [ "opacity-0"
                , "opacity-10"
                , "opacity-20"
                , "opacity-30"
                , "opacity-40"
                , "opacity-50"
                , "opacity-60"
                , "opacity-70"
                , "opacity-80"
                , "opacity-90"
                , "opacity-100"
                ]
                |> Array.get (bucket (0.5 * value))
                |> Maybe.withDefault "opacity-0"
    in
    Html.div
        [ Html.Attributes.class "w-[16px] h-[16px] bg-rose-600"
        , Html.Attributes.class "md:w-[24px] md:h-[24px]"
        , Html.Attributes.class "lg:w-[32px] lg:h-[32px]"
        , Html.Attributes.class "xl:w-[40px] xl:h-[40px]"
        , Html.Attributes.class opacity
        ]
        []


bucket : Float -> Int
bucket value =
    -- Clamp to ensure value is in [0.0, 1.0]
    let
        clamped =
            clamp 0.0 1.0 value

        -- Multiply by 10 and floor
        -- Special case: if value is 1.0, we want bucket 9 not 10
        out =
            if clamped >= 1.0 then
                9

            else
                floor (clamped * 10)
    in
    out


viewSaeActivations : Maybe ( String, Int ) -> Requests.Requested (Dict.Dict String (List SaeActivation)) -> Html.Html Msg
viewSaeActivations focusedLatent requestedActivations =
    case requestedActivations of
        Requests.Initial ->
            Html.p
                [ Html.Attributes.class "italic" ]
                [ Html.text "Load an image to see SAE activations." ]

        Requests.Loading ->
            Html.p
                []
                [ Html.text "Loading SAE activations..." ]

        Requests.Loaded activations ->
            Html.div
                []
                (Dict.toList activations
                    |> List.map (uncurry (viewModelSaeActivations focusedLatent))
                )

        Requests.Failed err ->
            viewErr err


viewModelSaeActivations : Maybe ( String, Int ) -> String -> List SaeActivation -> Html.Html Msg
viewModelSaeActivations focusedLatent model saeActivations =
    Html.div
        []
        [ Html.p
            []
            [ Html.text model ]
        , Html.div
            [ Html.Attributes.class "grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3" ]
            (List.map
                (viewSaeActivation focusedLatent)
                saeActivations
            )
        ]


viewSaeActivation : Maybe ( String, Int ) -> SaeActivation -> Html.Html Msg
viewSaeActivation focusedLatent { model, latent, activations, examples } =
    let
        active =
            case focusedLatent of
                Nothing ->
                    False

                Just ( name, l ) ->
                    name == model && l == latent
    in
    Html.div []
        [ Html.details
            [ Html.Attributes.class "cursor-pointer rounded-lg border border-gray-200 bg-white shadow-sm hover:shadow-md transition-shadow duration-200 dark:border-gray-700 dark:bg-gray-800"
            , Html.Attributes.attribute "open" ""
            , Html.Events.onMouseEnter (FocusLatent model latent)
            , Html.Events.onMouseLeave (BlurLatent model latent)
            ]
            [ Html.summary
                [ Html.Attributes.class "cursor-pointer select-none px-4 py-3 hover:bg-gray-50 text-gray-900" ]
                [ Html.span
                    []
                    [ Html.text ("Latent: " ++ String.fromInt latent) ]
                ]
            , Html.div
                [ Html.Attributes.class "p-1 text-gray-600 border-t border-gray-200 grid gap-1 grid-cols-2 md:grid-cols-4" ]
                (List.map (viewHighlightedExample active) examples)
            ]
        ]


viewHighlightedExample : Bool -> HighlightedExample -> Html.Html Msg
viewHighlightedExample active { original, highlighted, label, exampleId } =
    let
        ( classOriginal, classHighlighted ) =
            if active then
                ( "opacity-0", "opacity-100" )

            else
                ( "opacity-100", "opacity-0" )
    in
    Html.div
        [ Html.Attributes.class "relative"
        , Html.Events.onClick (SelectExample exampleId)
        ]
        [ Html.img
            [ Html.Attributes.class "transition-opacity duration-100"
            , Html.Attributes.class classOriginal
            , Html.Attributes.src (Gradio.base64ImageToString original)
            ]
            []
        , Html.img
            [ Html.Attributes.class "absolute inset-0 transition-opacity duration-100"
            , Html.Attributes.class classHighlighted
            , Html.Attributes.src (Gradio.base64ImageToString highlighted)
            ]
            []
        ]


viewErr : String -> Html.Html Msg
viewErr err =
    Html.div
        [ Html.Attributes.class "relative rounded-lg border border-red-200 bg-red-50 p-4 m-4" ]
        [ Html.button
            []
            []
        , Html.h3
            [ Html.Attributes.class "font-bold text-red-800" ]
            [ Html.text "Error" ]
        , Html.p
            [ Html.Attributes.class "text-red-700" ]
            [ Html.text err ]
        ]



-- HELPERS


uncurry : (a -> b -> c) -> ( a, b ) -> c
uncurry f ( a, b ) =
    f a b



-- CONSTANTS


clipImagenetLatents : List Int
clipImagenetLatents =
    [ 4988
    ]
